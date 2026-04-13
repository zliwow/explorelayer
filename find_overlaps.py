#!/usr/bin/env python3
"""
Pad/RDL Overlap Detector — find what geometries sit under pad and RDL regions.

Reads a GDSII file plus a layer config JSON, flattens the cell hierarchy,
and reports every geometry on the "check" layers that overlaps with pad or
RDL polygons. Output is structured JSON suitable for report_generator.py.

Usage:
    python find_overlaps.py path/to/chip.gds --config layer_config.json
    python find_overlaps.py path/to/chip.gds --config layer_config.json --output results.json
    python find_overlaps.py path/to/chip.gds --config layer_config.json --cell CELLNAME
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import gdstk
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from shapely.validation import make_valid


def load_config(config_path):
    """Load and validate the layer config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)

    required = ["pad_layers", "check_layers"]
    for key in required:
        if key not in cfg:
            print(f"Error: config missing required key '{key}'")
            sys.exit(1)

    # Normalise pad_layers / rdl_layers to list of (layer, datatype) tuples
    cfg["pad_layers"] = [tuple(l) for l in cfg["pad_layers"]]
    cfg["rdl_layers"] = [tuple(l) for l in cfg.get("rdl_layers", [])]

    # Normalise check_layers
    for cl in cfg["check_layers"]:
        cl["layer"] = tuple(cl["layer"])

    cfg.setdefault("min_overlap_area", 0.0)
    return cfg


def flatten_cell(lib, cell_name=None):
    """
    Return a flattened copy of the target cell.

    Flattening resolves all nested references so every polygon is in
    absolute top-level coordinates — required before overlap detection.
    """
    if cell_name:
        cell = None
        for c in lib.cells:
            if c.name == cell_name:
                cell = c
                break
        if cell is None:
            print(f"Error: cell '{cell_name}' not found.")
            sys.exit(1)
    else:
        top = lib.top_level()
        if not top:
            print("Error: no top-level cells found.")
            sys.exit(1)
        cell = top[0]
        if len(top) > 1:
            print(f"Note: using top-level cell '{cell.name}'")

    flat = cell.copy(cell.name + "_flat_overlap")
    flat.flatten()
    return flat


def extract_polygons_by_layer(flat_cell):
    """
    Extract all polygons from a flattened cell grouped by (layer, datatype).

    Returns dict: (layer, datatype) -> list of shapely Polygons.
    """
    layer_polys = defaultdict(list)

    def add_poly(pts, layer, datatype):
        if len(pts) < 3:
            return
        try:
            sp = ShapelyPolygon(pts)
            if not sp.is_valid:
                sp = make_valid(sp)
            if sp.is_empty or sp.area == 0:
                return
            layer_polys[(layer, datatype)].append(sp)
        except Exception:
            pass  # skip degenerate geometry

    for poly in flat_cell.polygons:
        add_poly(poly.points, poly.layer, poly.datatype)

    for path in flat_cell.paths:
        for poly in path.to_polygons():
            add_poly(poly.points, poly.layer, poly.datatype)

    return dict(layer_polys)


def find_overlaps(layer_polys, cfg):
    """
    Core overlap detection.

    For every pad (and optionally RDL) polygon, check each "check" layer
    for intersecting geometries. Uses an R-tree spatial index per check
    layer for performance on large files.

    Returns a list of findings (dicts).
    """
    min_area = cfg["min_overlap_area"]

    # Gather pad polygons
    pad_polys = []
    for ldt in cfg["pad_layers"]:
        pad_polys.extend(layer_polys.get(ldt, []))

    # Gather RDL polygons
    rdl_polys = []
    for ldt in cfg["rdl_layers"]:
        rdl_polys.extend(layer_polys.get(ldt, []))

    # Combine into region list with a type tag
    regions = [(p, "pad") for p in pad_polys] + [(p, "rdl") for p in rdl_polys]

    if not regions:
        print("Warning: no pad or RDL polygons found on the configured layers.")
        return []

    # Build spatial indexes for each check layer
    check_indexes = {}
    for cl in cfg["check_layers"]:
        ldt = cl["layer"]
        polys = layer_polys.get(ldt, [])
        if polys:
            tree = STRtree(polys)
            check_indexes[cl["name"]] = (ldt, polys, tree)

    findings = []
    total_regions = len(regions)

    for idx, (region_poly, region_type) in enumerate(regions, 1):
        if idx % 50 == 0 or idx == total_regions:
            print(f"  Processing region {idx}/{total_regions} ...", end="\r")

        region_bb = region_poly.bounds  # (minx, miny, maxx, maxy)
        region_area = region_poly.area

        for layer_name, (ldt, polys, tree) in check_indexes.items():
            # Query the spatial index for candidate overlaps
            # shapely 2.x returns a numpy array of integer indices
            candidate_indices = tree.query(region_poly)
            for ci in candidate_indices:
                geom = polys[int(ci)]

                if not region_poly.intersects(geom):
                    continue

                try:
                    intersection = region_poly.intersection(geom)
                except Exception:
                    continue

                overlap_area = intersection.area
                if overlap_area < min_area:
                    continue

                geom_bb = geom.bounds
                findings.append({
                    "region_type": region_type,
                    "region_bounds": list(region_bb),
                    "region_area": round(region_area, 4),
                    "check_layer": layer_name,
                    "check_layer_number": list(ldt),
                    "overlap_area": round(overlap_area, 4),
                    "overlap_pct_of_region": round(100.0 * overlap_area / region_area, 2) if region_area > 0 else 0,
                    "overlap_pct_of_geometry": round(100.0 * overlap_area / geom.area, 2) if geom.area > 0 else 0,
                    "geometry_bounds": list(geom_bb),
                    "geometry_area": round(geom.area, 4),
                })

    print()  # clear the \r line
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Detect geometries overlapping with pad/RDL regions in a GDSII file."
    )
    parser.add_argument("gds_file", help="Path to the .gds file")
    parser.add_argument("--config", required=True, help="Path to layer_config.json")
    parser.add_argument("--cell", default=None, help="Target cell (default: top-level)")
    parser.add_argument("--output", default=None, help="Write JSON results to this file (default: stdout)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Load GDS
    print(f"Loading {args.gds_file} ...")
    t0 = time.time()
    lib = gdstk.read_gds(args.gds_file)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(lib.cells)} cells")

    # Flatten
    print("Flattening cell hierarchy ...")
    t0 = time.time()
    flat = flatten_cell(lib, args.cell)
    print(f"  Flattened in {time.time() - t0:.1f}s")

    # Extract polygons
    print("Extracting polygons by layer ...")
    layer_polys = extract_polygons_by_layer(flat)
    total = sum(len(v) for v in layer_polys.values())
    print(f"  {total} polygons across {len(layer_polys)} layer/datatype pairs")

    # Detect overlaps
    print("Detecting overlaps ...")
    t0 = time.time()
    findings = find_overlaps(layer_polys, cfg)
    print(f"  Found {len(findings)} overlaps in {time.time() - t0:.1f}s")

    # Build result payload
    result = {
        "gds_file": os.path.basename(args.gds_file),
        "cell": flat.name.replace("_flat_overlap", ""),
        "config": {
            "pad_layers": [list(l) for l in cfg["pad_layers"]],
            "rdl_layers": [list(l) for l in cfg["rdl_layers"]],
            "check_layers": [{"name": c["name"], "layer": list(c["layer"])} for c in cfg["check_layers"]],
            "min_overlap_area": cfg["min_overlap_area"],
        },
        "summary": {
            "total_pad_regions": sum(len(layer_polys.get(tuple(l), [])) for l in cfg["pad_layers"]),
            "total_rdl_regions": sum(len(layer_polys.get(tuple(l), [])) for l in cfg["rdl_layers"]),
            "total_overlaps": len(findings),
        },
        "findings": findings,
    }

    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results written to {args.output}")
    else:
        print("\n" + output_json)


if __name__ == "__main__":
    main()
