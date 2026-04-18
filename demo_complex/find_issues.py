#!/usr/bin/env python3
"""
Generalized aggressor->victim overlap detector.

For every (aggressor_layer, victim_layer) rule in the config, find every
overlapping polygon pair and resolve the victim back to its owning leaf
cell instance so the report can name the device (Q1, Mi_p, R_fb_top, ...).

Output: findings.json with one entry per overlap, grouped by
(rule, instance_key) and ranked by sensitivity.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import gdstk
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from shapely.validation import make_valid


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}


def load_cfg(path):
    with open(path) as f:
        cfg = json.load(f)
    for r in cfg["rules"]:
        r["aggressor"] = tuple(r["aggressor"])
        r["victim"] = tuple(r["victim"])
    cfg.setdefault("min_overlap_area", 0.0)
    return cfg


def walk_cell_instances(top_cell):
    """Yield (instance_path, leaf_cell, origin) for every leaf instance."""
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


def gather_all_cells(top_cell):
    out = {top_cell.name: top_cell}
    def _go(c):
        for ref in c.references:
            ch = ref.cell
            if isinstance(ch, gdstk.Cell) and ch.name not in out:
                out[ch.name] = ch
                _go(ch)
    _go(top_cell)
    return out


def extract_aggressor_polys(top_cell, layer_dt):
    """Aggressor polygons in top-cell coords (flattened)."""
    polys = []
    layer, dt = layer_dt
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


def extract_victim_polys_with_context(top_cell, all_cells, victim_layers):
    """
    {victim_layer_dt: [(shapely_poly, instance_path), ...]}

    Walks hierarchy so we can attribute each victim polygon to its leaf
    cell instance (BJT_PNP under BANDGAP_REF, etc.).
    """
    leaf_polys = defaultdict(lambda: defaultdict(list))
    target = set(victim_layers)
    for cname, cell in all_cells.items():
        for poly in cell.polygons:
            ld = (poly.layer, poly.datatype)
            if ld in target:
                leaf_polys[cname][ld].append(poly.points)

    by_layer = defaultdict(list)
    for path, leaf, origin in walk_cell_instances(top_cell):
        if leaf.name not in leaf_polys:
            continue
        for ld, plist in leaf_polys[leaf.name].items():
            for pts in plist:
                shifted = [(float(px) + origin[0], float(py) + origin[1])
                           for px, py in pts]
                sp = ShapelyPolygon(shifted)
                if not sp.is_valid:
                    sp = make_valid(sp)
                if sp.is_empty or sp.area == 0:
                    continue
                by_layer[ld].append((sp, path))
    return by_layer


def label_near(top_cell, x, y, radius=80.0):
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
    return " / ".join(p[0] for p in path)


def instance_key(path):
    parts = []
    for name, origin in path[1:]:
        parts.append(f"{name}@({origin[0]:.0f},{origin[1]:.0f})")
    return "/".join(parts) if parts else "top"


def block_name(path):
    """Top-of-hierarchy block name (e.g. BANDGAP_REF)."""
    if len(path) >= 2:
        return path[1][0]
    return path[0][0]


def find_overlaps(gds_path, cfg_path, out_path):
    cfg = load_cfg(cfg_path)
    print(f"Loading {gds_path} ...")
    lib = gdstk.read_gds(gds_path)
    tops = lib.top_level()
    if not tops:
        sys.exit("No top-level cell")
    top = tops[0]
    print(f"Top cell: {top.name}")

    all_cells = gather_all_cells(top)

    victim_layers = sorted({r["victim"] for r in cfg["rules"]})
    victim_data = extract_victim_polys_with_context(top, all_cells, victim_layers)
    for ld, lst in victim_data.items():
        print(f"  victim {ld}: {len(lst)} polys")

    # Build STRtree per victim layer once
    victim_index = {}
    for ld, lst in victim_data.items():
        if not lst:
            continue
        victim_index[ld] = (lst, STRtree([sp for sp, _ in lst]))

    findings = []
    for rule in cfg["rules"]:
        agg_polys = extract_aggressor_polys(top, rule["aggressor"])
        if not agg_polys:
            continue
        idx = victim_index.get(rule["victim"])
        if idx is None:
            continue
        lst, tree = idx
        for ai, agg in enumerate(agg_polys):
            agg_label = label_near(top, agg.centroid.x, agg.centroid.y)
            for tidx in tree.query(agg):
                sp, path = lst[int(tidx)]
                if not agg.intersects(sp):
                    continue
                inter = agg.intersection(sp)
                area = inter.area
                if area < cfg["min_overlap_area"]:
                    continue
                findings.append({
                    "rule": rule["name"],
                    "sensitivity": rule["sensitivity"],
                    "why": rule["why"],
                    "aggressor_label": rule["aggressor_label"],
                    "victim_label": rule["victim_label"],
                    "agg_index": ai,
                    "agg_route_label": agg_label,
                    "agg_bounds": [float(v) for v in agg.bounds],
                    "victim_bounds": [float(v) for v in sp.bounds],
                    "instance_path": path_label(path),
                    "instance_key": instance_key(path),
                    "block": block_name(path),
                    "overlap_area": round(area, 3),
                    "overlap_bounds": [float(v) for v in inter.bounds],
                })

    # Group by (rule, block) so the noisy rdl-over-diff fires inside a
    # block cluster up into a single finding card.
    grouped = defaultdict(lambda: {"count": 0, "total_area": 0.0, "items": []})
    for f in findings:
        key = (f["rule"], f["block"])
        g = grouped[key]
        g["count"] += 1
        g["total_area"] += f["overlap_area"]
        g["items"].append(f)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda kv: (SEVERITY_ORDER.get(kv[1]["items"][0]["sensitivity"], 9),
                        -kv[1]["total_area"])
    )

    report = {
        "gds_file": os.path.basename(gds_path),
        "top_cell": top.name,
        "finding_count": len(findings),
        "groups": [
            {
                "rule": k[0],
                "block": k[1],
                "sensitivity": v["items"][0]["sensitivity"],
                "why": v["items"][0]["why"],
                "aggressor_label": v["items"][0]["aggressor_label"],
                "victim_label": v["items"][0]["victim_label"],
                "agg_route_label": v["items"][0]["agg_route_label"],
                "instance_path": v["items"][0]["instance_path"],
                "count": v["count"],
                "total_overlap_area": round(v["total_area"], 3),
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
    print(f"Groups (rule x block):")
    for k, v in sorted_groups:
        items = v["items"]
        print(f"  [{items[0]['sensitivity']:8s}] {k[0]:30s}  in {k[1]:15s}  "
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
