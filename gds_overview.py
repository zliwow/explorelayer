#!/usr/bin/env python3
"""
gds_overview.py — one comprehensive markdown report answering David's
two questions (decomposed into six) from a GDS file alone.

  Q1. Can you identify top layer / any layer / bump / pad area?
      1. What's in this GDS?           (chip basics)
      2. What layers exist?            (full layer inventory)
      3. Which layer is the pad/bump?  (pad candidates, auto-picked)

  Q2. Can you identify devices under/near pad to first order?
      4. What device layers exist?     (diff/poly/BJT-marker candidates)
      5. Where are the pads?           (list with locations + sizes)
      6. What's under/near each pad?   (per-pad attribution)

Everything is auto-detected from the GDS. Layer identities are classified
by geometry (polygon count, size distribution, rectangularity, proximity
to die edge). The hierarchy walk composes rotations, mirrors, and
repetitions properly so owner attribution on real chips reaches the
actual leaf cell, not just the package wrapper.

Usage:
    python gds_overview.py chip.gds
    python gds_overview.py chip.gds --output overview.md
    python gds_overview.py chip.gds --pad-layer 60/0 --device-layers 1/0,5/0
    python gds_overview.py chip.gds --only-pads-with-hits
    python gds_overview.py chip.gds --near-radius 10 --sample 2000
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import gdstk
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon, box as shapely_box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid


# ── Affine transform composition (rotation / mirror / scale / translate) ─

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # a b c d e f → [x';y'] = [a b;c d][x;y] + [e;f]


def ref_transform(ref, offset=(0.0, 0.0)):
    """
    Compose a gdstk.Reference's transform (with optional repetition offset).
    Order: scale → x_reflection (y→−y) → rotate → translate.
    """
    mag = ref.magnification if ref.magnification is not None else 1.0
    rot = ref.rotation if ref.rotation is not None else 0.0
    flip = bool(ref.x_reflection)
    ox = float(ref.origin[0]) + float(offset[0])
    oy = float(ref.origin[1]) + float(offset[1])

    sx = mag
    sy = -mag if flip else mag
    cr = math.cos(rot)
    sr = math.sin(rot)

    # M = T(ox,oy) · R(rot) · diag(sx, sy)
    return (cr * sx, -sr * sy,
            sr * sx,  cr * sy,
            ox,       oy)


def compose(parent, child):
    """parent ∘ child: apply child first (inner), then parent (outer)."""
    pa, pb, pc, pd, pe, pf = parent
    ca, cb, cc, cd, ce, cf = child
    return (
        pa * ca + pb * cc,
        pa * cb + pb * cd,
        pc * ca + pd * cc,
        pc * cb + pd * cd,
        pa * ce + pb * cf + pe,
        pc * ce + pd * cf + pf,
    )


def apply_point(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + b * y + e, c * x + d * y + f)


def transform_bbox(m, bb):
    """Transform (x0,y0)-(x1,y1) by matrix m; return axis-aligned bbox."""
    (x0, y0), (x1, y1) = bb
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    xs, ys = [], []
    for x, y in corners:
        tx, ty = apply_point(m, x, y)
        xs.append(tx); ys.append(ty)
    return (min(xs), min(ys), max(xs), max(ys))


# ── Hierarchy walk with transform-aware bboxes ──────────────────────────

def walk_instance_bboxes(top_cell):
    """
    Return [(shapely_bbox, path_tuple, depth)] for every instance under
    top_cell, with rotation / mirror / scale / repetition all composed.

    Depth guard (64) breaks cycles.
    """
    entries = []
    stack = [(top_cell, IDENTITY, (top_cell.name,))]

    while stack:
        cell, transform, path = stack.pop()

        bb = cell.bounding_box()
        if bb is not None:
            (xa, ya, xb, yb) = transform_bbox(transform, bb)
            if xb > xa and yb > ya:
                entries.append((shapely_box(xa, ya, xb, yb), path, len(path)))

        for ref in cell.references:
            child = ref.cell
            if not isinstance(child, gdstk.Cell):
                continue
            rep = getattr(ref, "repetition", None)
            if rep is not None:
                try:
                    offsets = list(rep.offsets)
                except Exception:
                    offsets = [(0.0, 0.0)]
            else:
                offsets = [(0.0, 0.0)]
            for ox, oy in offsets:
                child_t = ref_transform(ref, (ox, oy))
                global_t = compose(transform, child_t)
                new_path = path + (child.name,)
                if len(new_path) > 64:
                    continue
                stack.append((child, global_t, new_path))

    return entries


def resolve_owner(point_xy, bbox_entries, bbox_tree):
    """Deepest bbox that contains point_xy, or None."""
    pt = Point(point_xy[0], point_xy[1])
    best_depth = -1
    best_path = None
    for ci in bbox_tree.query(pt):
        bbox_poly, path, depth = bbox_entries[int(ci)]
        if bbox_poly.contains(pt) and depth > best_depth:
            best_depth = depth
            best_path = path
    return best_path


def resolve_container(pad_shape, bbox_entries, bbox_tree):
    """Deepest bbox that fully CONTAINS pad_shape, or None."""
    best_depth = -1
    best_path = None
    for ci in bbox_tree.query(pad_shape):
        bbox_poly, path, depth = bbox_entries[int(ci)]
        if bbox_poly.contains(pad_shape) and depth > best_depth:
            best_depth = depth
            best_path = path
    return best_path


# ── Layer discovery + per-layer geometry stats ──────────────────────────

def discover_layers(lib):
    """Every (layer, datatype) seen in any cell."""
    seen = set()
    for c in lib.cells:
        for p in c.polygons:
            seen.add((p.layer, p.datatype))
        for path in c.paths:
            for p in path.to_polygons():
                seen.add((p.layer, p.datatype))
    return sorted(seen)


def layer_stats(cell, ldt, die_bb, die_area, sample_limit):
    """
    Geometry stats for one layer. If polygon count > sample_limit, stats
    are computed from a deterministic sample — total_count stays true.
    """
    polys = cell.get_polygons(layer=ldt[0], datatype=ldt[1])
    total = len(polys)
    if total == 0:
        return None

    if sample_limit and total > sample_limit:
        idx = np.linspace(0, total - 1, sample_limit, dtype=int)
        sample = [polys[i] for i in idx]
    else:
        sample = polys

    (dx0, dy0), (dx1, dy1) = die_bb
    dw = dx1 - dx0
    dh = dy1 - dy0
    edge_band = 0.15 * min(dw, dh)

    areas, ws, hs, ars, edges, is_rect = [], [], [], [], [], []
    for gp in sample:
        pts = gp.points if isinstance(gp, gdstk.Polygon) else gp
        if len(pts) < 3:
            continue
        pts = np.asarray(pts, dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w, h = x1 - x0, y1 - y0
        a = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        if a < 1e-12 or w <= 0 or h <= 0:
            continue
        areas.append(a)
        ws.append(w); hs.append(h)
        ars.append(max(w, h) / max(min(w, h), 1e-9))
        edges.append(min(x0 - dx0, dx1 - x1, y0 - dy0, dy1 - y1))
        is_rect.append((a / (w * h) > 0.9) and len(pts) <= 6)

    if not areas:
        return {"total_count": total, "sample_count": 0}

    areas = np.array(areas); edges = np.array(edges); is_rect = np.array(is_rect)
    # Total area extrapolated from sample
    est_total_area = float(np.sum(areas) * (total / len(areas)))

    return {
        "total_count": total,
        "sample_count": len(areas),
        "total_area": est_total_area,
        "area_pct_of_die": (est_total_area / die_area * 100.0) if die_area > 0 else 0.0,
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "max_area": float(np.max(areas)),
        "area_cv": float(np.std(areas) / max(np.mean(areas), 1e-9)),
        "mean_w": float(np.mean(ws)),
        "mean_h": float(np.mean(hs)),
        "median_ar": float(np.median(ars)),
        "pct_rect": float(np.mean(is_rect) * 100.0),
        "pct_near_edge": float(np.mean(edges < edge_band) * 100.0),
    }


def classify_layer(s):
    """Category + confidence + reasoning, from geometry stats."""
    if s is None or s.get("sample_count", 0) == 0:
        return "empty", 0, "no usable polygons"
    n = s["total_count"]; ma = s["mean_area"]; cv = s["area_cv"]
    pr = s["pct_rect"]; pe = s["pct_near_edge"]; ar = s["median_ar"]
    ap = s["area_pct_of_die"]

    if n <= 2 and ap > 80:
        return "die_boundary", 95, f"1-2 polys covering {ap:.0f}% of die"
    if n > 100000 and cv < 0.5 and ma < 50:
        return "fill", 90, f"{n/1e6:.1f}M uniform tiny polys (CV={cv:.2f})"
    if n > 1000000 and ap < 10:
        return "fill", 80, f"{n/1e6:.1f}M polys, only {ap:.1f}% of die"
    if 20 < n < 2000 and ma > 5000 and pr > 60 and pe > 40 and cv < 1.5:
        conf = 70
        if pe > 60: conf += 10
        if pr > 80: conf += 5
        if cv < 0.8: conf += 5
        return "pad_candidate", min(conf, 95), \
               f"{pe:.0f}% near edge, {pr:.0f}% rect, CV={cv:.2f}"
    if n < 3000 and ma > 10000 and ap > 50:
        return "top_metal_or_rdl", 65, f"{n} polys, {ap:.0f}% die coverage"
    if n < 5000 and ma > 1000 and pr > 50 and pe > 30 and ap < 30:
        return "passivation_or_pad", 50, f"medium rects, {pe:.0f}% near edge"
    if 1000 < n < 1000000 and 10 < ma < 50000:
        return "metal_routing", 50, f"{n} polys, mean area {ma:.0f}"
    if n < 50000 and ap > 20:
        return "implant_or_well", 45, f"{n} polys, {ap:.0f}% coverage"
    if 1000 < n < 500000 and ar > 3 and ma < 5000:
        return "poly_candidate", 50, f"thin AR={ar:.1f}, mean area {ma:.0f}"
    if 10000 < n < 1000000 and ma < 100 and cv < 1.0:
        return "via_or_contact", 55, f"{n} small uniform polys"
    if n < 20 and ap < 0.1:
        return "marker_or_text", 40, f"{n} polys, negligible area"
    if n < 50 and pe > 80 and ap > 1:
        return "seal_ring", 55, f"{n} polys, {pe:.0f}% at edge"
    return "unclassified", 20, f"{n} polys, mean area {ma:.0f}, {ap:.1f}% die"


# ── Auto-pick pad + device layers ───────────────────────────────────────

def pick_pad_layer(layer_infos, explicit):
    if explicit:
        return explicit, "user-specified"
    # Highest-confidence pad_candidate wins; else passivation_or_pad; else top_metal_or_rdl.
    for cat in ("pad_candidate", "passivation_or_pad", "top_metal_or_rdl"):
        cands = [li for li in layer_infos if li["category"] == cat]
        if cands:
            cands.sort(key=lambda li: -li["confidence"])
            return cands[0]["ldt"], f"auto-picked ({cat}, {cands[0]['confidence']}% confidence)"
    return None, "no candidate found"


def pick_device_layers(layer_infos, explicit):
    if explicit:
        return explicit, "user-specified"
    # Heuristic: anything that looks like poly, implant, or a generic device
    # layer (not fill, not pad, not metal, not unknown).
    out = []
    for li in layer_infos:
        cat = li["category"]
        if cat in ("poly_candidate", "implant_or_well"):
            out.append(li["ldt"])
    return out, "auto-picked (poly + implant/well layers)"


# ── Pad extraction + merging ────────────────────────────────────────────

def extract_polygons(cell, ldt):
    out = []
    for gp in cell.get_polygons(layer=ldt[0], datatype=ldt[1]):
        pts = gp.points if isinstance(gp, gdstk.Polygon) else gp
        if len(pts) < 3:
            continue
        sp = ShapelyPolygon(pts)
        if not sp.is_valid:
            sp = make_valid(sp)
        if sp.is_empty or sp.area == 0:
            continue
        out.append(sp)
    return out


def merge_pads(pad_polys):
    if not pad_polys:
        return []
    merged = unary_union(pad_polys)
    parts = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
    items = []
    for p in parts:
        if p.is_empty:
            continue
        cx, cy = p.centroid.x, p.centroid.y
        x0, y0, x1, y1 = p.bounds
        items.append({"shape": p, "center": (cx, cy),
                      "w": x1 - x0, "h": y1 - y0, "bounds": (x0, y0, x1, y1)})
    items.sort(key=lambda it: (it["center"][0], it["center"][1]))
    return items


def scan_pads(pads, device_index, bbox_entries, bbox_tree, near_radius):
    """For each pad, aggregate (owner_path, ldt) → under_area / near_distance."""
    for pad in pads:
        under = defaultdict(float)
        near = {}
        seen_under = set()
        buffered = pad["shape"].buffer(near_radius) if near_radius > 0 else pad["shape"]

        for ldt, polys, tree in device_index:
            for ci in tree.query(pad["shape"]):
                geom = polys[int(ci)]
                if not pad["shape"].intersects(geom):
                    continue
                try:
                    a = pad["shape"].intersection(geom).area
                except Exception:
                    continue
                if a <= 0:
                    continue
                owner = resolve_owner((geom.centroid.x, geom.centroid.y),
                                      bbox_entries, bbox_tree)
                owner_str = "/".join(owner) if owner else "(unresolved)"
                key = (owner_str, ldt)
                under[key] += a
                seen_under.add(key)

            if near_radius > 0:
                for ci in tree.query(buffered):
                    geom = polys[int(ci)]
                    if pad["shape"].intersects(geom):
                        continue
                    if not buffered.intersects(geom):
                        continue
                    d = pad["shape"].distance(geom)
                    if d > near_radius:
                        continue
                    owner = resolve_owner((geom.centroid.x, geom.centroid.y),
                                          bbox_entries, bbox_tree)
                    owner_str = "/".join(owner) if owner else "(unresolved)"
                    key = (owner_str, ldt)
                    if key in seen_under:
                        continue
                    if key not in near or d < near[key]:
                        near[key] = d

        pad["under"] = dict(under)
        pad["near"] = near
    return pads


# ── Markdown rendering ──────────────────────────────────────────────────

def layer_label(ldt, infos):
    cat = next((li["category"] for li in infos if li["ldt"] == ldt), None)
    return f"{ldt[0]}/{ldt[1]} ({cat})" if cat else f"{ldt[0]}/{ldt[1]}"


def render_markdown(d):
    L = []
    L.append(f"# GDS overview — {d['gds_file']}\n")
    L.append(f"_Generated in {d['elapsed']:.1f}s. "
             f"Answers David's two questions, decomposed into six sections._\n")

    # ─── 1. Chip basics ─────────────────────────────────────────────────
    L.append("## 1. What's in this GDS?\n")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| Top cell | `{d['top_cell']}` |")
    L.append(f"| Cells in library | {d['n_cells']} |")
    L.append(f"| Die bounding box | ({d['die_bb'][0]:.1f}, {d['die_bb'][1]:.1f}) → "
             f"({d['die_bb'][2]:.1f}, {d['die_bb'][3]:.1f}) |")
    L.append(f"| Die size | {d['die_w']:.1f} × {d['die_h']:.1f} µm "
             f"(area {d['die_area']:.0f} µm²) |")
    L.append(f"| Hierarchy max depth | {d['max_depth']} levels |")
    L.append(f"| Unique layer/datatype pairs | {len(d['layer_infos'])} |")
    L.append(f"| Total polygons (all layers) | {d['total_polys']:,} |")
    L.append("")

    # ─── 2. Layer inventory ─────────────────────────────────────────────
    L.append("## 2. What layers exist?\n")
    L.append("Every layer/datatype pair in the file, sorted by category "
             "(pad candidates first). Categories are heuristic from geometry — "
             "confirm with the designer if ambiguous.\n")
    L.append("| Layer | Category | Conf. | Polys | Die % | Mean area (µm²) | "
             "% rect | % near edge | Reasoning |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for li in d["layer_infos"]:
        s = li.get("stats") or {}
        L.append("| {l}/{dt} | {cat} | {conf}% | {n} | {ap:.1f}% | {ma} | "
                 "{pr} | {pe} | {r} |".format(
                    l=li["ldt"][0], dt=li["ldt"][1],
                    cat=li["category"],
                    conf=li["confidence"],
                    n=s.get("total_count", 0),
                    ap=s.get("area_pct_of_die", 0.0),
                    ma=f"{s.get('mean_area', 0):.1f}" if s else "—",
                    pr=f"{s.get('pct_rect', 0):.0f}" if s else "—",
                    pe=f"{s.get('pct_near_edge', 0):.0f}" if s else "—",
                    r=li["reasoning"],
                 ))
    L.append("")

    # ─── 3. Pad/bump identification ─────────────────────────────────────
    L.append("## 3. Which layer is the pad / bump?\n")
    L.append(f"**Chosen pad layer:** `{d['pad_layer'][0]}/{d['pad_layer'][1]}` — {d['pad_pick_reason']}\n")
    pad_cands = [li for li in d["layer_infos"]
                 if li["category"] in ("pad_candidate", "passivation_or_pad")]
    if pad_cands:
        L.append("**All pad-like candidates (in order of confidence):**\n")
        L.append("| Layer | Category | Conf. | Polys | Mean area (µm²) | % near edge |")
        L.append("|---|---|---:|---:|---:|---:|")
        pad_cands.sort(key=lambda li: -li["confidence"])
        for li in pad_cands:
            s = li.get("stats") or {}
            L.append(f"| {li['ldt'][0]}/{li['ldt'][1]} | {li['category']} | "
                     f"{li['confidence']}% | {s.get('total_count', 0)} | "
                     f"{s.get('mean_area', 0):.1f} | {s.get('pct_near_edge', 0):.0f}% |")
    else:
        L.append("_No pad_candidate layers auto-detected. "
                 "The designer may know the correct layer number; pass it via "
                 "`--pad-layer NN/DT`._")
    L.append("")

    # ─── 4. Device layers ───────────────────────────────────────────────
    L.append("## 4. What device layers exist?\n")
    L.append(f"**Chosen device layers:** "
             + ", ".join(f"`{l[0]}/{l[1]}`" for l in d["device_layers"])
             + f" — {d['device_pick_reason']}\n")
    dev_cands = [li for li in d["layer_infos"]
                 if li["category"] in ("poly_candidate", "implant_or_well",
                                       "via_or_contact")]
    if dev_cands:
        L.append("**Device-layer candidates in this GDS:**\n")
        L.append("| Layer | Category | Conf. | Polys | Reasoning |")
        L.append("|---|---|---:|---:|---|")
        for li in dev_cands:
            s = li.get("stats") or {}
            L.append(f"| {li['ldt'][0]}/{li['ldt'][1]} | {li['category']} | "
                     f"{li['confidence']}% | {s.get('total_count', 0)} | "
                     f"{li['reasoning']} |")
    else:
        L.append("_No device-like candidates auto-detected. "
                 "Pass explicit layers via `--device-layers a/b,c/d`._")
    L.append("")

    # ─── 5. Pad listing ─────────────────────────────────────────────────
    L.append("## 5. Where are the pads?\n")
    L.append(f"**{len(d['pads'])} pad area(s) found on layer "
             f"{d['pad_layer'][0]}/{d['pad_layer'][1]}.**")
    if d['pads']:
        with_hits = sum(1 for p in d['pads'] if p["under"] or p["near"])
        L.append(f"{with_hits} have device overlap or a nearby device "
                 f"(buffer {d['near_radius']:.1f} µm); "
                 f"{len(d['pads']) - with_hits} are clean.\n")
    L.append("")
    # Summary pad table
    if d['pads']:
        L.append("| # | Center (µm) | Size (µm) | Container cell | Under? | Near? |")
        L.append("|---:|---|---|---|---:|---:|")
        for i, p in enumerate(d['pads'], 1):
            under_n = sum(1 for _ in p["under"])
            near_n = sum(1 for _ in p["near"])
            L.append(f"| {i} | ({p['center'][0]:.1f}, {p['center'][1]:.1f}) "
                     f"| {p['w']:.1f} × {p['h']:.1f} | `{p['container']}` "
                     f"| {under_n} | {near_n} |")
    L.append("")

    # ─── 6. Per-pad devices under / near ────────────────────────────────
    L.append("## 6. What's under / near each pad?\n")
    pads_to_show = [p for p in d["pads"] if (p["under"] or p["near"])] \
                   if d["only_with_hits"] else d["pads"]
    if d["only_with_hits"]:
        L.append(f"_Showing only pads with hits ({len(pads_to_show)} of "
                 f"{len(d['pads'])}). Run without `--only-pads-with-hits` for full list._\n")

    for i, p in enumerate(pads_to_show, 1):
        real_i = d["pads"].index(p) + 1
        L.append(f"### PAD #{real_i} — at ({p['center'][0]:.1f}, {p['center'][1]:.1f}), "
                 f"{p['w']:.1f} × {p['h']:.1f} µm")
        L.append(f"Container cell: `{p['container']}`\n")

        L.append("**Directly under:**\n")
        if p["under"]:
            L.append("| Cell (owner path) | Layer | Overlap area (µm²) |")
            L.append("|---|---|---:|")
            for (own, ldt), a in sorted(p["under"].items(), key=lambda kv: -kv[1]):
                L.append(f"| `{own}` | {layer_label(ldt, d['layer_infos'])} | {a:.1f} |")
        else:
            L.append("_Nothing under._")
        L.append("")

        L.append(f"**Within {d['near_radius']:.1f} µm:**\n")
        if p["near"]:
            L.append("| Cell (owner path) | Layer | Min distance (µm) |")
            L.append("|---|---|---:|")
            for (own, ldt), dist in sorted(p["near"].items(), key=lambda kv: kv[1]):
                L.append(f"| `{own}` | {layer_label(ldt, d['layer_infos'])} | {dist:.2f} |")
        else:
            L.append("_Nothing within buffer._")
        L.append("\n---\n")

    return "\n".join(L)


# ── Main ────────────────────────────────────────────────────────────────

def parse_layer(s):
    if "/" in s:
        l, dt = s.split("/", 1)
        return (int(l), int(dt))
    return (int(s), 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("gds_file")
    p.add_argument("--output", default=None, help="Default: overview.md next to GDS")
    p.add_argument("--pad-layer", default=None,
                   help="Force pad layer NN/DT. Default: auto-pick.")
    p.add_argument("--device-layers", default=None,
                   help="Force device layers a/b,c/d,... Default: auto-pick.")
    p.add_argument("--near-radius", type=float, default=5.0,
                   help="Buffer (µm) for 'near' check. Default: 5.")
    p.add_argument("--sample", type=int, default=1000,
                   help="Polygons sampled per layer for stats. Default: 1000. "
                        "Stats converge well below this; pad scan uses ALL polys.")
    p.add_argument("--only-pads-with-hits", action="store_true",
                   help="In section 6, hide pads with no under/near device.")
    p.add_argument("--cell", default=None, help="Override top cell name.")
    args = p.parse_args()

    t_start = time.time()

    print(f"[1/8] Loading {args.gds_file} ...")
    t0 = time.time()
    lib = gdstk.read_gds(args.gds_file)
    print(f"      {len(lib.cells)} cells in {time.time()-t0:.1f}s")

    if args.cell:
        cell = next((c for c in lib.cells if c.name == args.cell), None)
        if cell is None:
            sys.exit(f"Cell '{args.cell}' not found.")
    else:
        tops = lib.top_level()
        if not tops:
            sys.exit("No top-level cells.")
        cell = tops[0]
    bb = cell.bounding_box()
    if bb is None:
        sys.exit(f"Top cell '{cell.name}' has no bounding box.")
    die_bb = bb
    die_w = bb[1][0] - bb[0][0]
    die_h = bb[1][1] - bb[0][1]
    die_area = die_w * die_h
    print(f"      top cell: {cell.name}   die {die_w:.1f} × {die_h:.1f} µm")

    print("[2/8] Discovering layers ...")
    t0 = time.time()
    all_layers = discover_layers(lib)
    print(f"      {len(all_layers)} layer/datatype pairs in {time.time()-t0:.1f}s")

    print("[3/8] Classifying each layer (this is the slow step) ...")
    t0 = time.time()
    layer_infos = []
    total_polys = 0
    for i, ldt in enumerate(all_layers, 1):
        print(f"      [{i}/{len(all_layers)}] layer {ldt[0]}/{ldt[1]} ...",
              end="", flush=True)
        st = layer_stats(cell, ldt, die_bb, die_area, args.sample)
        cat, conf, reason = classify_layer(st)
        if st:
            total_polys += st.get("total_count", 0)
        layer_infos.append({
            "ldt": ldt,
            "category": cat,
            "confidence": conf,
            "reasoning": reason,
            "stats": st,
        })
        print(f" {cat} ({conf}%)")
    print(f"      classified in {time.time()-t0:.1f}s")

    # Sort by priority so the inventory table reads nicely
    priority = ["pad_candidate", "passivation_or_pad", "top_metal_or_rdl",
                "poly_candidate", "implant_or_well", "metal_routing",
                "via_or_contact", "die_boundary", "seal_ring",
                "marker_or_text", "fill", "unclassified", "empty"]
    pmap = {c: i for i, c in enumerate(priority)}
    layer_infos.sort(key=lambda li: (pmap.get(li["category"], 99), -li["confidence"]))

    print("[4/8] Choosing pad + device layers ...")
    pad_layer, pad_reason = pick_pad_layer(
        layer_infos, parse_layer(args.pad_layer) if args.pad_layer else None)
    if args.device_layers:
        device_layers = [parse_layer(x) for x in args.device_layers.split(",")]
        device_reason = "user-specified"
    else:
        device_layers, device_reason = pick_device_layers(layer_infos, None)
    if pad_layer is None:
        sys.exit("No pad layer could be auto-detected. Pass --pad-layer NN/DT.")
    print(f"      pad: {pad_layer[0]}/{pad_layer[1]}  devices: "
          + (", ".join(f"{l[0]}/{l[1]}" for l in device_layers) or "(none)"))

    print("[5/8] Walking hierarchy (transform-aware) ...")
    t0 = time.time()
    bbox_entries = walk_instance_bboxes(cell)
    max_depth = max((d for _, _, d in bbox_entries), default=1)
    bbox_tree = STRtree([e[0] for e in bbox_entries])
    print(f"      {len(bbox_entries)} instance bboxes  max depth {max_depth}  "
          f"{time.time()-t0:.1f}s")

    print("[6/8] Extracting pads ...")
    t0 = time.time()
    pad_polys = extract_polygons(cell, pad_layer)
    pads = merge_pads(pad_polys)
    for p in pads:
        own = resolve_container(p["shape"], bbox_entries, bbox_tree)
        p["container"] = "/".join(own) if own else "(spans multiple cells)"
    print(f"      {len(pad_polys)} raw → {len(pads)} merged pad areas  "
          f"{time.time()-t0:.1f}s")

    print("[7/8] Indexing device layers ...")
    t0 = time.time()
    device_index = []
    for ldt in device_layers:
        polys = extract_polygons(cell, ldt)
        if polys:
            device_index.append((ldt, polys, STRtree(polys)))
        print(f"      {ldt[0]}/{ldt[1]}: {len(polys)} polys")
    print(f"      indexed in {time.time()-t0:.1f}s")

    print(f"[8/8] Scanning {len(pads)} pads against {len(device_index)} device layer(s) ...")
    t0 = time.time()
    scan_pads(pads, device_index, bbox_entries, bbox_tree, args.near_radius)
    print(f"      scanned in {time.time()-t0:.1f}s")

    data = {
        "gds_file": os.path.basename(args.gds_file),
        "top_cell": cell.name,
        "n_cells": len(lib.cells),
        "die_bb": (die_bb[0][0], die_bb[0][1], die_bb[1][0], die_bb[1][1]),
        "die_w": die_w, "die_h": die_h, "die_area": die_area,
        "max_depth": max_depth,
        "layer_infos": layer_infos,
        "total_polys": total_polys,
        "pad_layer": pad_layer,
        "pad_pick_reason": pad_reason,
        "device_layers": device_layers,
        "device_pick_reason": device_reason,
        "near_radius": args.near_radius,
        "pads": pads,
        "only_with_hits": args.only_pads_with_hits,
        "elapsed": time.time() - t_start,
    }

    md = render_markdown(data)
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.gds_file)), "overview.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nDone. Wrote {out_path}  ({data['elapsed']:.1f}s total)")


if __name__ == "__main__":
    main()
