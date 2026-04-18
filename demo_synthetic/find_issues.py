#!/usr/bin/env python3
"""
RDL-over-sensitive detector.

For every RDL polygon in the top cell, find every overlapping polygon
on the configured 'check layers'. Resolve each overlap to the owning
cell instance (BJT_PNP, MOS_PMOS, etc.) so downstream reporting can
name the matched-pair devices (Q1, Q2, ...) directly.

Output: findings.json with one entry per overlap, grouped by cell
instance and ranked by sensitivity.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import gdstk
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.strtree import STRtree
from shapely.validation import make_valid


def load_cfg(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg["pad_layers"]  = [tuple(l) for l in cfg.get("pad_layers", [])]
    cfg["rdl_layers"]  = [tuple(l) for l in cfg.get("rdl_layers", [])]
    for cl in cfg["check_layers"]:
        cl["layer"] = tuple(cl["layer"])
    cfg.setdefault("min_overlap_area", 0.0)
    cfg.setdefault("sensitivity_ranking", {})
    return cfg


def walk_cell_instances(top_cell, parent_xform=(0.0, 0.0, 0.0, False)):
    """
    Yield (instance_path, leaf_cell, (dx, dy, rot, xrefl)) for every
    leaf-cell instance reached from top_cell.

    instance_path is a list of (cell_name, origin) tuples tracing the hierarchy.
    """
    stack = [(top_cell, (0.0, 0.0), [(top_cell.name, (0.0, 0.0))])]
    while stack:
        cell, origin, path = stack.pop()
        has_refs = False
        for ref in cell.references:
            child = ref.cell
            if isinstance(child, gdstk.Cell):
                has_refs = True
                ref_origin = (origin[0] + float(ref.origin[0]),
                              origin[1] + float(ref.origin[1]))
                stack.append((child, ref_origin, path + [(child.name, ref_origin)]))
        if not has_refs:
            yield path, cell, origin


def extract_rdl_polygons(top_cell, rdl_layers):
    """Get RDL polygons in top-cell coordinates (flattened)."""
    polys = []
    for layer, dt in rdl_layers:
        for p in top_cell.get_polygons(layer=layer, datatype=dt):
            pts = p.points if isinstance(p, gdstk.Polygon) else p
            if len(pts) < 3:
                continue
            sp = ShapelyPolygon(pts)
            if not sp.is_valid:
                sp = make_valid(sp)
            if not sp.is_empty and sp.area > 0:
                polys.append(sp)
    return polys


def extract_check_polys_with_context(top_cell, check_layers):
    """
    For every (layer, dt) in check_layers, flatten the hierarchy down to
    leaf instances and record each polygon tagged with its owning instance
    chain.

    Returns dict keyed by the check-layer name:
       {check_name: [(shapely_polygon, instance_path), ...]}
    """
    # Build instance index once: which leaf cells are reachable, and for each
    # leaf we want to know polygons per target layer (in leaf-local coords),
    # and then transform into global coords per instance.
    by_layer = {cl["name"]: [] for cl in check_layers}

    # Collect leaf polygons per (layer,dt), keyed by cell name
    leaf_polys = defaultdict(lambda: defaultdict(list))  # cell_name -> (layer,dt) -> [points]
    for layer, dt in {tuple(cl["layer"]) for cl in check_layers}:
        # Walk every cell in the library — easier than per-cell get_polygons
        pass

    # We need access to the library to iterate cells. Use top_cell.dependencies.
    deps = set()
    def _collect(c):
        for ref in c.references:
            ch = ref.cell
            if isinstance(ch, gdstk.Cell) and ch.name not in deps:
                deps.add(ch.name)
                _collect(ch)
    _collect(top_cell)
    all_cells = {top_cell.name: top_cell}
    def _gather_cells(c):
        for ref in c.references:
            ch = ref.cell
            if isinstance(ch, gdstk.Cell) and ch.name not in all_cells:
                all_cells[ch.name] = ch
                _gather_cells(ch)
    _gather_cells(top_cell)

    target_lds = {tuple(cl["layer"]): cl["name"] for cl in check_layers}

    for cname, cell in all_cells.items():
        for poly in cell.polygons:
            ld = (poly.layer, poly.datatype)
            if ld in target_lds:
                leaf_polys[cname][ld].append(poly.points)

    # Now walk hierarchy and attach instance paths.
    for path, leaf, origin in walk_cell_instances(top_cell):
        if leaf.name not in leaf_polys:
            continue
        for ld, plist in leaf_polys[leaf.name].items():
            cname = target_lds[ld]
            for pts in plist:
                # Apply instance origin (we assume no rotation in our demo)
                shifted = [(float(px) + origin[0], float(py) + origin[1]) for px, py in pts]
                sp = ShapelyPolygon(shifted)
                if not sp.is_valid:
                    sp = make_valid(sp)
                if sp.is_empty or sp.area == 0:
                    continue
                by_layer[cname].append((sp, path))
    return by_layer


def label_near(top_cell, x, y, radius=10.0):
    """Return the label text closest to (x, y) in the top cell within radius, else None."""
    best = None
    best_d = radius ** 2
    for lbl in top_cell.labels:
        lx, ly = float(lbl.origin[0]), float(lbl.origin[1])
        d = (lx - x) ** 2 + (ly - y) ** 2
        if d < best_d:
            best_d = d
            best = lbl.text
    return best


def path_label(path):
    """Render an instance path like 'analog_chip_top / BANDGAP_REF / BJT_PNP'."""
    return " / ".join(p[0] for p in path)


def instance_key(path):
    """
    Build a stable, human-friendly key for grouping overlaps per device
    instance: e.g. 'BANDGAP_REF@(50,80)/BJT_PNP@(70,100)'.
    """
    parts = []
    for name, origin in path[1:]:  # skip top
        parts.append(f"{name}@({origin[0]:.0f},{origin[1]:.0f})")
    return "/".join(parts) if parts else "top"


def find_overlaps(gds_path, cfg_path, out_path):
    cfg = load_cfg(cfg_path)
    print(f"Loading {gds_path} ...")
    lib = gdstk.read_gds(gds_path)
    top_cells = lib.top_level()
    if not top_cells:
        print("No top-level cell.")
        sys.exit(1)
    top = top_cells[0]
    print(f"Top cell: {top.name}")

    rdl_polys = extract_rdl_polygons(top, cfg["rdl_layers"])
    print(f"RDL polygons: {len(rdl_polys)}")
    if not rdl_polys:
        print("No RDL found — nothing to check.")
        return

    check_data = extract_check_polys_with_context(top, cfg["check_layers"])
    for name, lst in check_data.items():
        print(f"  check layer {name}: {len(lst)} polys")

    # Spatial indexes per check layer
    indexes = {}
    for name, lst in check_data.items():
        if not lst:
            continue
        geoms = [sp for sp, _ in lst]
        indexes[name] = (lst, STRtree(geoms))

    findings = []
    for ri, rdl in enumerate(rdl_polys):
        rdl_label = label_near(top, rdl.centroid.x, rdl.centroid.y, radius=50)
        for name, (lst, tree) in indexes.items():
            for idx in tree.query(rdl):
                sp, path = lst[int(idx)]
                if not rdl.intersects(sp):
                    continue
                inter = rdl.intersection(sp)
                area = inter.area
                if area < cfg["min_overlap_area"]:
                    continue
                findings.append({
                    "rdl_index": ri,
                    "rdl_label": rdl_label,
                    "rdl_bounds": [float(v) for v in rdl.bounds],
                    "check_layer": name,
                    "sensitivity": cfg["sensitivity_ranking"].get(name, "unknown"),
                    "instance_path": path_label(path),
                    "instance_key": instance_key(path),
                    "overlap_area": round(area, 3),
                    "overlap_bounds": [float(v) for v in inter.bounds],
                })

    # Group by (instance_key, check_layer) so matched-pair devices cluster up nicely
    grouped = defaultdict(lambda: {"count": 0, "total_area": 0.0, "items": []})
    for f in findings:
        key = (f["instance_key"], f["check_layer"])
        g = grouped[key]
        g["count"] += 1
        g["total_area"] += f["overlap_area"]
        g["items"].append(f)

    # Sort groups by sensitivity severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
    sorted_groups = sorted(
        grouped.items(),
        key=lambda kv: (severity_order.get(kv[1]["items"][0]["sensitivity"], 9),
                        -kv[1]["total_area"])
    )

    report = {
        "gds_file": os.path.basename(gds_path),
        "top_cell": top.name,
        "rdl_polygon_count": len(rdl_polys),
        "finding_count": len(findings),
        "groups": [
            {
                "instance_key": k[0],
                "check_layer": k[1],
                "sensitivity": v["items"][0]["sensitivity"],
                "count": v["count"],
                "total_overlap_area": round(v["total_area"], 3),
                "instance_path": v["items"][0]["instance_path"],
                "items": v["items"],
            }
            for k, v in sorted_groups
        ],
        "raw_findings": findings,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== Summary ===")
    print(f"Total overlaps: {len(findings)}")
    print(f"Groups (by device instance × check layer):")
    for k, v in sorted_groups[:20]:
        items = v["items"]
        print(f"  [{items[0]['sensitivity']:8s}] {k[0]:50s}  {k[1]:10s}  "
              f"count={v['count']}  area={v['total_area']:.1f}")
    print(f"\nFull report -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gds")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="findings.json")
    args = p.parse_args()
    find_overlaps(args.gds, args.config, args.output)


if __name__ == "__main__":
    main()
