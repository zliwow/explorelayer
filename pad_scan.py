#!/usr/bin/env python3
"""
pad_scan.py — GDS-only pad-area + devices-nearby scan.

Answers two questions directly on a GDS file, no netlist required:
  1. Where are the pads / top-layer / bump areas?
  2. Which cells/devices sit directly under, or within N µm of, each pad?

Hierarchy-aware: each nearby shape is attributed to its owning cell
instance path (e.g. TOP/ESD_IO/M_clamp), so the reviewer sees WHAT is
under the pad, not just polygon coordinates.

Usage:
    python pad_scan.py chip.gds
    python pad_scan.py chip.gds --pad-layer 60/0
    python pad_scan.py chip.gds --pad-layer 50/0 \\
        --sensitive-layers 1/0,5/0,82/5 --near-radius 5
    python pad_scan.py chip.gds --output pad_scan.md
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import gdstk
from shapely.geometry import Point, Polygon as ShapelyPolygon, box as shapely_box
from shapely.strtree import STRtree
from shapely.validation import make_valid


# Sensible defaults for the demo layer stack. For a real chip the user
# should pass --pad-layer / --sensitive-layers / --layer-names.
DEFAULT_PAD_CANDIDATES = [(60, 0), (50, 0)]  # pad opening first, then RDL
DEFAULT_SENSITIVE_LAYERS = [(1, 0), (5, 0), (82, 5)]  # diff, poly, bjt_marker
DEFAULT_LAYER_NAMES = {
    (1, 0):  "diff",
    (2, 0):  "nwell",
    (5, 0):  "poly",
    (6, 0):  "poly_fill",
    (10, 0): "contact",
    (11, 0): "met1",
    (22, 0): "met2",
    (33, 0): "met3",
    (40, 0): "met_thick",
    (50, 0): "rdl",
    (60, 0): "pad",
    (82, 5): "bjt_marker",
    (90, 0): "hi_z_marker",
}
DEFAULT_NEAR_RADIUS = 5.0


# ── Layer parsing ──────────────────────────────────────────────────────

def parse_layer(s):
    """'60/0' → (60, 0). Accepts 'NN' as (NN, 0)."""
    if "/" in s:
        l, d = s.split("/", 1)
        return (int(l), int(d))
    return (int(s), 0)


def parse_layer_list(s):
    return [parse_layer(x) for x in s.split(",") if x.strip()]


def layer_label(ldt, names):
    nm = names.get(ldt)
    return f"{nm} ({ldt[0]}/{ldt[1]})" if nm else f"{ldt[0]}/{ldt[1]}"


# ── Hierarchy walk (owner-path attribution, same pattern as v2) ────────

def walk_instance_bboxes(top_cell):
    """
    Return [(shapely_bbox, path_tuple, depth)] for every instance under
    top_cell. Translation-only composition — good enough for owner-name
    attribution, which is all we need here.

    Duplicate path tuples are allowed (e.g. two instances of BJT_PNP both
    have path (TOP, BANDGAP, BJT_PNP)) — we differentiate them by origin.
    Depth guard (len(path) > 64) is the cycle breaker.
    """
    entries = []
    stack = [(top_cell, (0.0, 0.0), (top_cell.name,))]

    while stack:
        cell, origin, path = stack.pop()

        bb = cell.bounding_box()
        if bb is not None:
            (x0, y0), (x1, y1) = bb
            x0 += origin[0]; x1 += origin[0]
            y0 += origin[1]; y1 += origin[1]
            if x1 > x0 and y1 > y0:
                entries.append((shapely_box(x0, y0, x1, y1), path, len(path)))

        for ref in cell.references:
            child = ref.cell
            if not isinstance(child, gdstk.Cell):
                continue
            dx = float(ref.origin[0])
            dy = float(ref.origin[1])
            rep = getattr(ref, "repetition", None)
            if rep is not None:
                try:
                    offsets = list(rep.offsets)
                except Exception:
                    offsets = [(0.0, 0.0)]
            else:
                offsets = [(0.0, 0.0)]
            for ox, oy in offsets:
                new_origin = (origin[0] + dx + ox, origin[1] + dy + oy)
                new_path = path + (child.name,)
                if len(new_path) > 64:
                    continue
                stack.append((child, new_origin, new_path))

    return entries


def resolve_owner(point_xy, bbox_entries, bbox_tree):
    """Deepest instance whose bbox contains point_xy, else None."""
    pt = Point(point_xy[0], point_xy[1])
    candidates = bbox_tree.query(pt)
    best_depth = -1
    best_path = None
    for ci in candidates:
        bbox_poly, path, depth = bbox_entries[int(ci)]
        if bbox_poly.contains(pt) and depth > best_depth:
            best_depth = depth
            best_path = path
    return best_path


# ── Polygon extraction ─────────────────────────────────────────────────

def extract_polygons(cell, ldt):
    """Flatten polygons for a single (layer, datatype), return shapely list."""
    out = []
    raw = cell.get_polygons(layer=ldt[0], datatype=ldt[1])
    for gp in raw:
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


def pick_pad_layer(cell, explicit, candidates):
    """Return (layer, datatype) — either the user's choice or first non-empty candidate."""
    if explicit:
        return explicit
    for ldt in candidates:
        polys = cell.get_polygons(layer=ldt[0], datatype=ldt[1])
        if polys:
            return ldt
    return None


# ── Merge nearby pads into contiguous "pad areas" ──────────────────────

def merge_pad_polygons(pad_polys):
    """
    A single pad is often drawn as several overlapping rectangles (opening
    + surrounding metal). Merge anything that touches, and return a list of
    (pad_shape, center_xy, width, height) sorted left-to-right, bottom-up.
    """
    if not pad_polys:
        return []
    from shapely.ops import unary_union
    merged = unary_union(pad_polys)
    if merged.geom_type == "Polygon":
        parts = [merged]
    else:
        parts = list(merged.geoms)

    items = []
    for p in parts:
        if p.is_empty:
            continue
        cx, cy = p.centroid.x, p.centroid.y
        x0, y0, x1, y1 = p.bounds
        items.append((p, (cx, cy), x1 - x0, y1 - y0))
    # left→right, then bottom→up
    items.sort(key=lambda it: (it[1][0], it[1][1]))
    return items


# ── The scan ───────────────────────────────────────────────────────────

def scan(pad_shapes, sensitive_index, bbox_entries, bbox_tree, near_radius):
    """
    For each pad return:
      {
        'under': {(owner_path, ldt): total_overlap_area},
        'near':  {(owner_path, ldt): min_distance},
      }
    'near' excludes anything already in 'under'.
    """
    results = []
    for pad_shape, center, w, h in pad_shapes:
        buffered = pad_shape.buffer(near_radius) if near_radius > 0 else pad_shape

        under = defaultdict(float)
        near = {}  # min distance
        seen_under_keys = set()

        for ldt, polys, tree in sensitive_index:
            # Direct overlap
            for ci in tree.query(pad_shape):
                geom = polys[int(ci)]
                if not pad_shape.intersects(geom):
                    continue
                try:
                    inter = pad_shape.intersection(geom)
                except Exception:
                    continue
                area = inter.area
                if area <= 0:
                    continue
                owner = resolve_owner((geom.centroid.x, geom.centroid.y),
                                      bbox_entries, bbox_tree)
                owner_str = "/".join(owner) if owner else "(unresolved)"
                key = (owner_str, ldt)
                under[key] += area
                seen_under_keys.add(key)

            # Near (within buffer, not already under)
            if near_radius > 0:
                for ci in tree.query(buffered):
                    geom = polys[int(ci)]
                    if pad_shape.intersects(geom):
                        continue  # already counted as "under"
                    if not buffered.intersects(geom):
                        continue
                    d = pad_shape.distance(geom)
                    if d > near_radius:
                        continue
                    owner = resolve_owner(
                        (geom.centroid.x, geom.centroid.y),
                        bbox_entries, bbox_tree,
                    )
                    owner_str = "/".join(owner) if owner else "(unresolved)"
                    key = (owner_str, ldt)
                    if key in seen_under_keys:
                        continue
                    if key not in near or d < near[key]:
                        near[key] = d

        results.append({
            "center": center,
            "width": w,
            "height": h,
            "bounds": pad_shape.bounds,
            "under": dict(under),
            "near": near,
        })
    return results


def locate_pad_owner(pad_shape, bbox_entries, bbox_tree):
    """
    Deepest instance whose bbox fully CONTAINS the pad polygon. Pads that
    span multiple cells (e.g. an RDL strip across several blocks) can't be
    attributed to a single owner — report that instead.
    """
    candidates = bbox_tree.query(pad_shape)
    best_depth = -1
    best_path = None
    for ci in candidates:
        bbox_poly, path, depth = bbox_entries[int(ci)]
        if bbox_poly.contains(pad_shape) and depth > best_depth:
            best_depth = depth
            best_path = path
    if best_path is None:
        return "(spans multiple cells — see 'under' table)"
    return "/".join(best_path)


# ── Markdown ───────────────────────────────────────────────────────────

def render_markdown(data):
    lines = []
    lines.append(f"# Pad scan — {data['gds_file']}\n")
    lines.append(f"Top cell: `{data['top_cell']}`  ")
    lines.append(f"Pad layer scanned: {layer_label(data['pad_layer'], data['layer_names'])}  ")
    lines.append(f"Sensitive layers: "
                 + ", ".join(layer_label(l, data['layer_names'])
                             for l in data['sensitive_layers']) + "  ")
    lines.append(f"Pads detected: {len(data['pads'])}  ")
    lines.append(f"\"Near\" radius: {data['near_radius']} µm\n")
    lines.append("---\n")

    if not data["pads"]:
        lines.append("_No pad polygons found on the requested layer._\n")
        return "\n".join(lines)

    for i, pad in enumerate(data["pads"], 1):
        cx, cy = pad["center"]
        lines.append(
            f"## PAD #{i} — at ({cx:.1f}, {cy:.1f}), "
            f"{pad['width']:.1f} × {pad['height']:.1f} µm"
        )
        lines.append(f"Located in cell: `{pad['owner']}`\n")

        lines.append("### Directly under this pad\n")
        if pad["under"]:
            lines.append("| Cell (owner path) | Layer | Overlap area (µm²) |")
            lines.append("|---|---|---:|")
            rows = sorted(pad["under"].items(),
                          key=lambda kv: -kv[1])
            for (owner, ldt), area in rows:
                lines.append(
                    f"| `{owner}` | {layer_label(ldt, data['layer_names'])} "
                    f"| {area:.1f} |"
                )
        else:
            lines.append("_Nothing on the sensitive layers sits under this pad._")
        lines.append("")

        lines.append(f"### Within {data['near_radius']} µm of this pad\n")
        if pad["near"]:
            lines.append("| Cell (owner path) | Layer | Min distance (µm) |")
            lines.append("|---|---|---:|")
            rows = sorted(pad["near"].items(), key=lambda kv: kv[1])
            for (owner, ldt), dist in rows:
                lines.append(
                    f"| `{owner}` | {layer_label(ldt, data['layer_names'])} "
                    f"| {dist:.2f} |"
                )
        else:
            lines.append("_Nothing within the buffer._")
        lines.append("\n---\n")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("gds_file")
    p.add_argument("--pad-layer", default=None,
                   help="Pad layer as NN/DT (e.g. 60/0). "
                        "Default: auto-pick first non-empty of 60/0, 50/0.")
    p.add_argument("--sensitive-layers", default=None,
                   help="Comma-separated NN/DT list (e.g. 1/0,5/0,82/5). "
                        "Default: diff, poly, bjt_marker.")
    p.add_argument("--near-radius", type=float, default=DEFAULT_NEAR_RADIUS,
                   help=f"Buffer (µm) around each pad for 'near'. "
                        f"Default: {DEFAULT_NEAR_RADIUS}")
    p.add_argument("--cell", default=None,
                   help="Top cell name. Default: first top-level cell.")
    p.add_argument("--output", default=None,
                   help="Markdown output path. Default: pad_scan.md next to GDS.")
    args = p.parse_args()

    explicit_pad = parse_layer(args.pad_layer) if args.pad_layer else None
    sensitive = (parse_layer_list(args.sensitive_layers)
                 if args.sensitive_layers else DEFAULT_SENSITIVE_LAYERS)

    print(f"Loading {args.gds_file} ...")
    t0 = time.time()
    lib = gdstk.read_gds(args.gds_file)
    print(f"  {len(lib.cells)} cells in {time.time() - t0:.1f}s")

    if args.cell:
        cell = next((c for c in lib.cells if c.name == args.cell), None)
        if cell is None:
            sys.exit(f"Cell '{args.cell}' not found.")
    else:
        top = lib.top_level()
        if not top:
            sys.exit("No top-level cells.")
        cell = top[0]
    print(f"Top cell: {cell.name}")

    pad_ldt = pick_pad_layer(cell, explicit_pad, DEFAULT_PAD_CANDIDATES)
    if pad_ldt is None:
        sys.exit("No pad polygons found (tried 60/0 and 50/0). "
                 "Pass --pad-layer NN/DT.")
    print(f"Pad layer: {pad_ldt[0]}/{pad_ldt[1]}")
    print(f"Sensitive layers: "
          + ", ".join(f"{l[0]}/{l[1]}" for l in sensitive))

    print("Extracting pad polygons ...")
    pad_polys = extract_polygons(cell, pad_ldt)
    print(f"  {len(pad_polys)} raw pad polys")

    pad_shapes = merge_pad_polygons(pad_polys)
    print(f"  merged into {len(pad_shapes)} pad areas")

    print("Extracting sensitive polygons ...")
    sensitive_index = []
    for ldt in sensitive:
        polys = extract_polygons(cell, ldt)
        print(f"  layer {ldt[0]}/{ldt[1]}: {len(polys)} polys")
        if polys:
            sensitive_index.append((ldt, polys, STRtree(polys)))

    print("Building hierarchy bbox index ...")
    bbox_entries = walk_instance_bboxes(cell)
    bbox_tree = STRtree([e[0] for e in bbox_entries])
    print(f"  {len(bbox_entries)} instance bboxes")

    print("Scanning ...")
    pads = scan(pad_shapes, sensitive_index, bbox_entries, bbox_tree,
                args.near_radius)

    # Attach owner-of-pad for each pad
    for pad, (shape, _, _, _) in zip(pads, pad_shapes):
        pad["owner"] = locate_pad_owner(shape, bbox_entries, bbox_tree)

    data = {
        "gds_file": os.path.basename(args.gds_file),
        "top_cell": cell.name,
        "pad_layer": pad_ldt,
        "sensitive_layers": sensitive,
        "layer_names": DEFAULT_LAYER_NAMES,
        "near_radius": args.near_radius,
        "pads": pads,
    }

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.gds_file)),
        "pad_scan.md",
    )
    md = render_markdown(data)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
