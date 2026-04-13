#!/usr/bin/env python3
"""
GDS Layout Explorer — CLI tool for inspecting GDSII files.

Prints file summary, layer info, cell hierarchy, bounding boxes,
and polygon statistics. Optionally exports per-layer SVGs.

Usage:
    python explore_gds.py path/to/chip.gds
    python explore_gds.py path/to/chip.gds --cell CELLNAME
    python explore_gds.py path/to/chip.gds --layers
    python explore_gds.py path/to/chip.gds --hierarchy
    python explore_gds.py path/to/chip.gds --export-layers output_dir/
"""

import argparse
import os
import sys
from collections import defaultdict

import gdstk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon


def load_library(gds_path):
    """Load a GDSII file and return the gdstk Library."""
    if not os.path.isfile(gds_path):
        print(f"Error: file not found: {gds_path}")
        sys.exit(1)
    print(f"Loading {gds_path} ...")
    lib = gdstk.read_gds(gds_path)
    print(f"Loaded library with {len(lib.cells)} cells.")
    return lib


def find_top_cells(lib):
    """Return list of top-level cells (cells not referenced by any other cell)."""
    return lib.top_level()


def get_cell_by_name(lib, name):
    """Look up a cell by name. Exit with error if not found."""
    for cell in lib.cells:
        if cell.name == name:
            return cell
    print(f"Error: cell '{name}' not found in library.")
    print("Available cells:", ", ".join(c.name for c in lib.cells))
    sys.exit(1)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(lib, cell):
    """Print high-level file and cell summary."""
    top_cells = find_top_cells(lib)
    print("\n=== FILE SUMMARY ===")
    print(f"  Total cells:      {len(lib.cells)}")
    print(f"  Top-level cells:  {len(top_cells)}")
    for tc in top_cells:
        print(f"    - {tc.name}")
    print(f"\n=== INSPECTING CELL: {cell.name} ===")
    bb = cell.bounding_box()
    if bb is not None:
        (x0, y0), (x1, y1) = bb
        print(f"  Bounding box: ({x0:.3f}, {y0:.3f}) to ({x1:.3f}, {y1:.3f})")
        print(f"  Width x Height: {x1 - x0:.3f} x {y1 - y0:.3f}")
    else:
        print("  Bounding box: (empty cell)")
    print(f"  Polygons:    {len(cell.polygons)}")
    print(f"  Paths:       {len(cell.paths)}")
    print(f"  References:  {len(cell.references)}")
    print(f"  Labels:      {len(cell.labels)}")


# ── Layer summary ────────────────────────────────────────────────────────────

def collect_layer_stats(cell, flatten=True):
    """
    Collect per-layer polygon count and total area.

    If flatten=True the cell is flattened first so nested geometries are
    included. Returns dict keyed by (layer, datatype).
    """
    if flatten:
        # Work on a copy so we don't mutate the original
        flat = cell.copy(cell.name + "_flat")
        flat.flatten()
    else:
        flat = cell

    stats = defaultdict(lambda: {"count": 0, "area": 0.0})

    for poly in flat.polygons:
        key = (poly.layer, poly.datatype)
        stats[key]["count"] += 1
        # Use shapely for robust area calculation
        pts = poly.points
        if len(pts) >= 3:
            try:
                sp = ShapelyPolygon(pts)
                if sp.is_valid:
                    stats[key]["area"] += sp.area
            except Exception:
                pass  # skip degenerate polygons

    # Also count polygons inside paths (gdstk paths are parameterized)
    for path in flat.paths:
        for poly in path.to_polygons():
            key = (poly.layer, poly.datatype)
            stats[key]["count"] += 1
            pts = poly.points
            if len(pts) >= 3:
                try:
                    sp = ShapelyPolygon(pts)
                    if sp.is_valid:
                        stats[key]["area"] += sp.area
                except Exception:
                    pass

    return dict(stats)


def print_layer_summary(cell):
    """Print per-layer polygon count and area."""
    stats = collect_layer_stats(cell, flatten=True)
    if not stats:
        print("\n  (no polygons found)")
        return
    print("\n=== LAYER SUMMARY (flattened) ===")
    print(f"  {'Layer':>6}  {'Datatype':>8}  {'Polygons':>9}  {'Total Area':>14}")
    print(f"  {'-----':>6}  {'--------':>8}  {'---------':>9}  {'----------':>14}")
    for (layer, dt) in sorted(stats.keys()):
        s = stats[(layer, dt)]
        print(f"  {layer:>6}  {dt:>8}  {s['count']:>9}  {s['area']:>14.2f}")
    print(f"\n  Unique layer/datatype pairs: {len(stats)}")


# ── Hierarchy ────────────────────────────────────────────────────────────────

def build_hierarchy(cell, depth=0, visited=None):
    """
    Recursively build a printable hierarchy tree.

    Returns a list of (indent_level, cell_name, instance_count) tuples.
    """
    if visited is None:
        visited = set()
    lines = []
    # Count how many times each child cell is referenced
    child_counts = defaultdict(int)
    for ref in cell.references:
        child_counts[ref.cell.name] += 1
    for child_name, count in sorted(child_counts.items()):
        tag = " (circular ref)" if child_name in visited else ""
        lines.append((depth, child_name, count, tag))
        if child_name not in visited:
            visited.add(child_name)
            # Find the actual cell object
            child_cell = ref.cell if isinstance(ref.cell, gdstk.Cell) else None
            # gdstk references hold the actual cell object
            for r in cell.references:
                if r.cell.name == child_name:
                    child_cell = r.cell
                    break
            if child_cell is not None:
                lines.extend(build_hierarchy(child_cell, depth + 1, visited))
    return lines


def print_hierarchy(cell):
    """Print cell hierarchy as an indented tree."""
    print(f"\n=== CELL HIERARCHY (root: {cell.name}) ===")
    lines = build_hierarchy(cell)
    if not lines:
        print("  (no sub-cell references)")
        return
    for depth, name, count, tag in lines:
        indent = "  " + "  " * depth
        count_str = f" x{count}" if count > 1 else ""
        print(f"{indent}├── {name}{count_str}{tag}")


# ── SVG export ───────────────────────────────────────────────────────────────

def export_layer_svgs(cell, output_dir):
    """
    Export one SVG per layer showing all polygons on that layer.

    Uses matplotlib to render polygons. Each SVG is saved as
    layer_<L>_<DT>.svg in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten to get absolute coordinates
    flat = cell.copy(cell.name + "_export_flat")
    flat.flatten()

    # Group polygons by layer
    layer_polys = defaultdict(list)
    for poly in flat.polygons:
        key = (poly.layer, poly.datatype)
        layer_polys[key].append(poly.points)
    for path in flat.paths:
        for poly in path.to_polygons():
            key = (poly.layer, poly.datatype)
            layer_polys[key].append(poly.points)

    if not layer_polys:
        print("  No polygons to export.")
        return

    bb = flat.bounding_box()
    if bb is None:
        print("  Empty bounding box, skipping export.")
        return
    (x0, y0), (x1, y1) = bb
    width = x1 - x0
    height = y1 - y0

    for (layer, dt), polys in sorted(layer_polys.items()):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8 * height / max(width, 1e-9)))
        patches = []
        for pts in polys:
            patches.append(mpatches.Polygon(pts, closed=True))
        pc = PatchCollection(patches, alpha=0.6, edgecolor="black", linewidth=0.3)
        ax.add_collection(pc)
        ax.set_xlim(x0 - width * 0.02, x1 + width * 0.02)
        ax.set_ylim(y0 - height * 0.02, y1 + height * 0.02)
        ax.set_aspect("equal")
        ax.set_title(f"Layer {layer} / Datatype {dt}  ({len(polys)} polygons)")
        fname = os.path.join(output_dir, f"layer_{layer}_{dt}.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore a GDSII file: layers, hierarchy, and statistics."
    )
    parser.add_argument("gds_file", help="Path to the .gds file")
    parser.add_argument("--cell", default=None, help="Inspect a specific cell instead of top-level")
    parser.add_argument("--layers", action="store_true", help="Only print layer summary")
    parser.add_argument("--hierarchy", action="store_true", help="Only print cell hierarchy tree")
    parser.add_argument("--export-layers", default=None, metavar="DIR",
                        help="Export each layer as a separate SVG to this directory")
    args = parser.parse_args()

    lib = load_library(args.gds_file)

    # Pick which cell to inspect
    if args.cell:
        cell = get_cell_by_name(lib, args.cell)
    else:
        top = find_top_cells(lib)
        if not top:
            print("Error: no top-level cells found.")
            sys.exit(1)
        cell = top[0]
        if len(top) > 1:
            print(f"Note: multiple top-level cells found. Using '{cell.name}'.")
            print(f"  Use --cell to pick another: {[c.name for c in top]}")

    # Decide what to print
    show_all = not args.layers and not args.hierarchy and not args.export_layers

    if show_all or not (args.layers or args.hierarchy or args.export_layers):
        print_summary(lib, cell)

    if show_all or args.layers:
        print_layer_summary(cell)

    if show_all or args.hierarchy:
        print_hierarchy(cell)

    if args.export_layers:
        print(f"\n=== EXPORTING LAYER SVGs to {args.export_layers} ===")
        export_layer_svgs(cell, args.export_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()
