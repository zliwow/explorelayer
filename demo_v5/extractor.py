#!/usr/bin/env python3
"""
extractor.py — stage 1 of the v5 pipeline.

Produces extraction.json from a GDS file. Deterministic, no LLM. Output
is the contract the stage-2 reasoner consumes.

What we extract:
  1. chip basics       top cell, die bbox, units, cell count
  2. layer inventory   per-layer geometry fingerprint + weak heuristic category
  3. layer-pair overlap    pairwise intersection area for non-fill layers
  4. cell hierarchy    top-N cells by polygon count, with bboxes + child names
  5. layer → cell map  which cells each layer appears in (top 20 per layer)
  6. labels            every text label grouped by layer/cell (if present)
  7. pad candidates    merged pad-like shapes on every pad-like layer

Stage 2 reads this JSON and assigns semantics (RDL? BJT marker? bandgap cell?).
Stage 1 never guesses at semantics; the weak "category" field in (2) is a
sanity hint only.

Usage:
    python demo_v5/extractor.py chip.gds
    python demo_v5/extractor.py chip.gds --output demo_v5/extraction.json
    python demo_v5/extractor.py chip.gds --sample 2000 --max-pair-layers 20
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import gdstk
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid


# ── Affine transform composition ────────────────────────────────────────

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def ref_transform(ref, offset=(0.0, 0.0)):
    mag = ref.magnification if ref.magnification is not None else 1.0
    rot = ref.rotation if ref.rotation is not None else 0.0
    flip = bool(ref.x_reflection)
    ox = float(ref.origin[0]) + float(offset[0])
    oy = float(ref.origin[1]) + float(offset[1])
    sx = mag
    sy = -mag if flip else mag
    cr = math.cos(rot); sr = math.sin(rot)
    return (cr * sx, -sr * sy, sr * sx, cr * sy, ox, oy)


def compose(parent, child):
    pa, pb, pc, pd, pe, pf = parent
    ca, cb, cc, cd, ce, cf = child
    return (pa*ca + pb*cc, pa*cb + pb*cd,
            pc*ca + pd*cc, pc*cb + pd*cd,
            pa*ce + pb*cf + pe, pc*ce + pd*cf + pf)


def apply_point(m, x, y):
    a, b, c, d, e, f = m
    return (a*x + b*y + e, c*x + d*y + f)


def transform_bbox(m, bb):
    (x0, y0), (x1, y1) = bb
    xs, ys = [], []
    for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        tx, ty = apply_point(m, x, y)
        xs.append(tx); ys.append(ty)
    return (min(xs), min(ys), max(xs), max(ys))


# ── Hierarchy walk ──────────────────────────────────────────────────────

def walk_instance_bboxes(top_cell):
    """Return [(shapely_bbox, path_tuple, depth)] under top_cell."""
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
            offsets = [(0.0, 0.0)]
            if rep is not None:
                try:
                    raw = rep.offsets
                    if raw is not None:
                        offsets = [(float(x), float(y)) for x, y in raw]
                except Exception:
                    offsets = [(0.0, 0.0)]
            for ox, oy in offsets:
                child_t = ref_transform(ref, (ox, oy))
                global_t = compose(transform, child_t)
                new_path = path + (child.name,)
                if len(new_path) > 64:
                    continue
                stack.append((child, global_t, new_path))
    return entries


# ── Layer discovery + stats ─────────────────────────────────────────────

def discover_layers(lib):
    seen = set()
    for c in lib.cells:
        for p in c.polygons:
            seen.add((p.layer, p.datatype))
        for path in c.paths:
            for p in path.to_polygons():
                seen.add((p.layer, p.datatype))
    return sorted(seen)


def layer_stats(cell, ldt, die_bb, die_area, sample_limit):
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

    areas, ws, hs, ars, edges, is_rect, centroids = [], [], [], [], [], [], []
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
        centroids.append(((x0 + x1) / 2, (y0 + y1) / 2))

    if not areas:
        return {"total_count": total, "sample_count": 0}

    areas = np.array(areas); edges = np.array(edges); is_rect = np.array(is_rect)
    est_total_area = float(np.sum(areas) * (total / len(areas)))

    return {
        "total_count": total,
        "sample_count": len(areas),
        "total_area_est": est_total_area,
        "area_pct_of_die": (est_total_area / die_area * 100.0) if die_area > 0 else 0.0,
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "max_area": float(np.max(areas)),
        "min_area": float(np.min(areas)),
        "area_cv": float(np.std(areas) / max(np.mean(areas), 1e-9)),
        "mean_w": float(np.mean(ws)),
        "mean_h": float(np.mean(hs)),
        "median_ar": float(np.median(ars)),
        "pct_rect": float(np.mean(is_rect) * 100.0),
        "pct_near_edge": float(np.mean(edges < edge_band) * 100.0),
    }


def weak_category(s):
    """
    Weak heuristic hint ONLY. Stage-2 LLM overrides this. Kept so the
    extraction JSON has a sanity baseline to compare against.
    """
    if s is None or s.get("sample_count", 0) == 0:
        return "empty", "no usable polygons"
    n = s["total_count"]; ma = s["mean_area"]; cv = s["area_cv"]
    pr = s["pct_rect"]; pe = s["pct_near_edge"]; ar = s["median_ar"]
    ap = s["area_pct_of_die"]
    if n <= 2 and ap > 80:
        return "die_boundary", f"1-2 polys covering {ap:.0f}% of die"
    if n > 100000 and cv < 0.5 and ma < 50:
        return "fill", f"{n/1e6:.1f}M uniform tiny polys"
    if n > 1000000 and ap < 10:
        return "fill", f"{n/1e6:.1f}M polys, only {ap:.1f}% of die"
    if 20 < n < 2000 and ma > 5000 and pr > 60 and pe > 40 and cv < 1.5:
        return "pad_like", f"{pe:.0f}% near edge, {pr:.0f}% rect"
    if n < 3000 and ma > 10000 and ap > 50:
        return "top_metal_like", f"{n} polys, {ap:.0f}% die coverage"
    if n < 5000 and ma > 1000 and pr > 50 and pe > 30 and ap < 30:
        return "pad_like", f"medium rects, {pe:.0f}% near edge"
    if 1000 < n < 1000000 and 10 < ma < 50000:
        return "metal_routing_like", f"{n} polys, mean area {ma:.0f}"
    if n < 50000 and ap > 20:
        return "implant_or_well_like", f"{n} polys, {ap:.0f}% coverage"
    if 1000 < n < 500000 and ar > 3 and ma < 5000:
        return "poly_like", f"thin AR={ar:.1f}, mean area {ma:.0f}"
    if 10000 < n < 1000000 and ma < 100 and cv < 1.0:
        return "via_like", f"{n} small uniform polys"
    if n < 20 and ap < 0.1:
        return "marker_like", f"{n} polys, negligible area"
    if n < 50 and pe > 80 and ap > 1:
        return "seal_ring_like", f"{n} polys, {pe:.0f}% at edge"
    return "unclassified", f"{n} polys, mean area {ma:.0f}"


# ── Layer → cell map ────────────────────────────────────────────────────

def layer_cell_map(lib, top_k=20):
    """For each layer, list the top-K cells by polygon count on that layer."""
    counts = defaultdict(lambda: defaultdict(int))
    for c in lib.cells:
        for p in c.polygons:
            counts[(p.layer, p.datatype)][c.name] += 1
    out = {}
    for ldt, d in counts.items():
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        out[f"{ldt[0]}/{ldt[1]}"] = [{"cell": k, "n_polys": v} for k, v in ranked[:top_k]]
    return out


# ── Labels ──────────────────────────────────────────────────────────────

def extract_labels(lib, cap_per_layer=100):
    """
    All labels grouped by layer/texttype. Layer names from the foundry
    (if the GDS carries them) usually live here.
    """
    out = defaultdict(list)
    for c in lib.cells:
        for lab in c.labels:
            out[(lab.layer, lab.texttype)].append({
                "text": lab.text,
                "cell": c.name,
                "xy": [float(lab.origin[0]), float(lab.origin[1])],
            })
    return {f"{ldt[0]}/{ldt[1]}": labs[:cap_per_layer] for ldt, labs in out.items()}


# ── Polygon extraction + pad merging ────────────────────────────────────

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


def merge_shapes(polys):
    if not polys:
        return []
    merged = unary_union(polys)
    parts = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
    items = []
    for p in parts:
        if p.is_empty or p.area == 0:
            continue
        cx, cy = p.centroid.x, p.centroid.y
        x0, y0, x1, y1 = p.bounds
        items.append({
            "center": [cx, cy],
            "w": x1 - x0, "h": y1 - y0,
            "area": p.area,
            "bounds": [x0, y0, x1, y1],
        })
    items.sort(key=lambda it: (it["center"][0], it["center"][1]))
    return items


# ── Pairwise layer overlap (the key new signal for semantics) ──────────

def layer_pair_overlaps(cell, layer_infos, max_layers=15, sample=500,
                        min_overlap_area=1.0):
    """
    For a subset of non-fill layers, compute pairwise intersection area.
    This is what lets stage 2 reason 'RDL sits on top of the pad layer',
    'BJT markers sit inside diff', etc.

    To stay tractable on huge GDS, only the top `max_layers` by 'interest'
    are paired. Interest = has polygons, not fill, not empty, not enormous.
    """
    candidates = []
    for li in layer_infos:
        s = li.get("stats") or {}
        n = s.get("total_count", 0)
        if n == 0 or n > 200000:
            continue
        if li["weak_category"] in ("fill", "empty"):
            continue
        candidates.append(li["ldt"])
    candidates = candidates[:max_layers]

    polys_by = {}
    trees_by = {}
    for ldt in candidates:
        all_polys = extract_polygons(cell, ldt)
        if not all_polys:
            continue
        if len(all_polys) > sample:
            idx = np.linspace(0, len(all_polys) - 1, sample, dtype=int)
            all_polys = [all_polys[i] for i in idx]
        polys_by[ldt] = all_polys
        trees_by[ldt] = STRtree(all_polys)

    out = []
    for i, a in enumerate(candidates):
        if a not in polys_by:
            continue
        a_area = sum(p.area for p in polys_by[a])
        for b in candidates[i+1:]:
            if b not in polys_by:
                continue
            total_overlap = 0.0
            for poly_a in polys_by[a]:
                for ci in trees_by[b].query(poly_a):
                    poly_b = polys_by[b][int(ci)]
                    if not poly_a.intersects(poly_b):
                        continue
                    try:
                        total_overlap += poly_a.intersection(poly_b).area
                    except Exception:
                        pass
            if total_overlap < min_overlap_area:
                continue
            b_area = sum(p.area for p in polys_by[b])
            out.append({
                "a": f"{a[0]}/{a[1]}",
                "b": f"{b[0]}/{b[1]}",
                "overlap_area": float(total_overlap),
                "a_pct": float(total_overlap / a_area * 100) if a_area > 0 else 0.0,
                "b_pct": float(total_overlap / b_area * 100) if b_area > 0 else 0.0,
                "note": "sampled" if any(
                    (li["ldt"] == a or li["ldt"] == b)
                    and (li.get("stats") or {}).get("total_count", 0) > sample
                    for li in layer_infos
                ) else "full",
            })
    out.sort(key=lambda x: -x["overlap_area"])
    return out


# ── Cell summary ────────────────────────────────────────────────────────

def cell_summary(lib, top_cell_name, bbox_entries, top_k=40):
    """
    Top-K cells by polygon count, with bbox (all instances bounded), list
    of child cell names, and a weak 'role hint' from name pattern.
    """
    # Polygon counts per cell (own geometry only; not including children)
    own_polys = {c.name: len(c.polygons) for c in lib.cells}

    # Aggregate instance bboxes per cell name
    bbox_by_cell = defaultdict(list)
    for sh, path, _depth in bbox_entries:
        x0, y0, x1, y1 = sh.bounds
        bbox_by_cell[path[-1]].append((x0, y0, x1, y1))

    # Child relationships
    children_by = {}
    for c in lib.cells:
        kids = {ref.cell.name for ref in c.references
                if isinstance(ref.cell, gdstk.Cell)}
        children_by[c.name] = sorted(kids)

    # Name-based hint (VERY weak — stage 2 LLM supersedes)
    def name_hint(n):
        u = n.upper()
        if any(t in u for t in ("BANDGAP", "BGR", "BG_", "_BG", "VBG", "VREF")):
            return "bandgap_like_name"
        if "LDO" in u or "REGULATOR" in u:
            return "ldo_like_name"
        if any(t in u for t in ("COMP", "COMPARATOR")):
            return "comparator_like_name"
        if "OSC" in u or "VCO" in u:
            return "oscillator_like_name"
        if "ESD" in u or "_IO" in u or "IO_" in u:
            return "io_or_esd_like_name"
        if "PAD" in u or "BUMP" in u:
            return "pad_like_name"
        return None

    ranked = sorted(lib.cells, key=lambda c: -own_polys[c.name])
    out = []
    for c in ranked[:top_k]:
        bbs = bbox_by_cell.get(c.name, [])
        agg_bb = None
        if bbs:
            agg_bb = [min(b[0] for b in bbs), min(b[1] for b in bbs),
                      max(b[2] for b in bbs), max(b[3] for b in bbs)]
        out.append({
            "name": c.name,
            "own_polys": own_polys[c.name],
            "n_instances": len(bbs),
            "children": children_by[c.name][:15],
            "n_children": len(children_by[c.name]),
            "bbox_any_instance": agg_bb,
            "name_hint": name_hint(c.name),
            "is_top": (c.name == top_cell_name),
        })
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("gds_file")
    ap.add_argument("--output", default=None,
                    help="Default: extraction.json next to the GDS")
    ap.add_argument("--sample", type=int, default=1000,
                    help="Polygons sampled per layer for stats/overlap. Default 1000.")
    ap.add_argument("--max-pair-layers", type=int, default=15,
                    help="Cap on layers used in pairwise overlap. Default 15.")
    ap.add_argument("--top-cells", type=int, default=40,
                    help="Top-K cells (by own polygon count) dumped. Default 40.")
    ap.add_argument("--cell", default=None, help="Override top cell.")
    args = ap.parse_args()

    t_start = time.time()

    print(f"[1/6] Loading {args.gds_file} ...")
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

    print("[2/6] Discovering + classifying layers ...")
    t0 = time.time()
    all_layers = discover_layers(lib)
    layer_infos = []
    total_polys = 0
    for i, ldt in enumerate(all_layers, 1):
        st = layer_stats(cell, ldt, die_bb, die_area, args.sample)
        cat, reason = weak_category(st)
        if st:
            total_polys += st.get("total_count", 0)
        layer_infos.append({
            "ldt": ldt,
            "layer": ldt[0], "datatype": ldt[1],
            "weak_category": cat,
            "weak_reasoning": reason,
            "stats": st,
        })
    print(f"      {len(all_layers)} layers classified in {time.time()-t0:.1f}s")

    print("[3/6] Walking hierarchy (transform-aware) ...")
    t0 = time.time()
    bbox_entries = walk_instance_bboxes(cell)
    max_depth = max((d for _, _, d in bbox_entries), default=1)
    print(f"      {len(bbox_entries)} instance bboxes  max depth {max_depth}  "
          f"{time.time()-t0:.1f}s")

    print("[4/6] Mapping layers → cells + extracting labels ...")
    t0 = time.time()
    lcmap = layer_cell_map(lib)
    labels = extract_labels(lib)
    print(f"      {sum(len(v) for v in labels.values())} labels across "
          f"{len(labels)} layers  {time.time()-t0:.1f}s")

    print(f"[5/6] Computing pairwise layer overlap (top {args.max_pair_layers} layers) ...")
    t0 = time.time()
    pairs = layer_pair_overlaps(cell, layer_infos,
                                max_layers=args.max_pair_layers,
                                sample=args.sample)
    print(f"      {len(pairs)} overlapping layer pairs  {time.time()-t0:.1f}s")

    print(f"[6/6] Summarizing top {args.top_cells} cells ...")
    t0 = time.time()
    cells = cell_summary(lib, cell.name, bbox_entries, top_k=args.top_cells)
    print(f"      done in {time.time()-t0:.1f}s")

    # Pad-like shapes on every pad-like layer (stage 2 decides which is "the" pad)
    pad_shapes_by_layer = {}
    for li in layer_infos:
        if li["weak_category"] not in ("pad_like", "top_metal_like"):
            continue
        polys = extract_polygons(cell, li["ldt"])
        merged = merge_shapes(polys)
        if merged:
            pad_shapes_by_layer[f"{li['ldt'][0]}/{li['ldt'][1]}"] = merged

    data = {
        "schema_version": 1,
        "gds_file": os.path.basename(args.gds_file),
        "gds_file_path": os.path.abspath(args.gds_file),
        "elapsed_s": round(time.time() - t_start, 2),
        "chip": {
            "top_cell": cell.name,
            "n_cells_in_library": len(lib.cells),
            "die_bbox": [die_bb[0][0], die_bb[0][1], die_bb[1][0], die_bb[1][1]],
            "die_w": die_w, "die_h": die_h, "die_area": die_area,
            "hierarchy_max_depth": max_depth,
            "n_instance_bboxes": len(bbox_entries),
            "units_um": True,  # gdstk reports in µm by default
            "total_polygons": total_polys,
        },
        "layers": [
            {
                "layer": li["layer"], "datatype": li["datatype"],
                "id": f"{li['layer']}/{li['datatype']}",
                "weak_category": li["weak_category"],
                "weak_reasoning": li["weak_reasoning"],
                "stats": li["stats"],
            } for li in layer_infos
        ],
        "layer_cell_map": lcmap,
        "labels_by_layer": labels,
        "layer_pair_overlaps": pairs,
        "cells": cells,
        "pad_shapes_by_layer": pad_shapes_by_layer,
    }

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.gds_file)), "extraction.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDone. Wrote {out_path}  ({size_kb:.1f} KB, {data['elapsed_s']}s total)")


if __name__ == "__main__":
    main()
