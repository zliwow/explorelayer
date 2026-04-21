#!/usr/bin/env python3
"""
Layer Auto-Identifier — deduce what each GDS layer represents by
analyzing polygon geometry patterns (size, shape, count, position).

No PDK or layer map needed. Works from the GDS file alone.

Outputs:
  - A classified layer table (JSON + human-readable)
  - Per-layer PNG thumbnails for visual confirmation

Usage:
    python identify_layers.py path/to/chip.gds --output layer_id_report/
    python identify_layers.py path/to/chip.gds --output layer_id_report/ --sample 500
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import gdstk
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def load_gds(path):
    print(f"Loading {path} ...")
    t0 = time.time()
    lib = gdstk.read_gds(path)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(lib.cells)} cells")
    top = lib.top_level()
    if not top:
        print("Error: no top-level cells.")
        sys.exit(1)
    cell = top[0]
    print(f"  Top cell: {cell.name}")
    bb = cell.bounding_box()
    if bb is not None:
        (x0, y0), (x1, y1) = bb
        die_area = (x1 - x0) * (y1 - y0)
        print(f"  Die: ({x0:.1f},{y0:.1f}) to ({x1:.1f},{y1:.1f}) — area {die_area:.0f}")
    else:
        die_area = 0
    return lib, cell, bb, die_area


def sample_polygons(cell, layer, datatype, max_sample=500):
    """
    Pull up to max_sample polygons from a layer using gdstk's C++
    get_polygons (handles hierarchy flattening internally).
    """
    all_polys = cell.get_polygons(layer=layer, datatype=datatype)
    total = len(all_polys)
    if total <= max_sample:
        sample = all_polys
    else:
        # Deterministic spread across the list
        indices = np.linspace(0, total - 1, max_sample, dtype=int)
        sample = [all_polys[i] for i in indices]
    return sample, total


def analyze_polygons(polys_sample, total_count, die_bb, die_area):
    """
    Compute geometry statistics from a sample of polygons.
    Returns a dict of features used for classification.
    """
    if not polys_sample:
        return None

    (die_x0, die_y0), (die_x1, die_y1) = die_bb
    die_w = die_x1 - die_x0
    die_h = die_y1 - die_y0

    areas = []
    widths = []
    heights = []
    aspect_ratios = []
    edge_distances = []  # min distance to die edge
    center_xs = []
    center_ys = []
    is_rect = []  # rectangle-like?

    for poly in polys_sample:
        pts = poly.points if isinstance(poly, gdstk.Polygon) else poly
        if len(pts) < 3:
            continue

        xs = pts[:, 0] if hasattr(pts, '__getitem__') and hasattr(pts[0], '__len__') else [p[0] for p in pts]
        ys = pts[:, 1] if hasattr(pts, '__getitem__') and hasattr(pts[0], '__len__') else [p[1] for p in pts]

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        w = maxx - minx
        h = maxy - miny

        # Signed area via shoelace
        n = len(xs)
        area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))

        if area < 1e-12:
            continue

        bbox_area = w * h if w > 0 and h > 0 else area
        rect_ratio = area / bbox_area if bbox_area > 0 else 0

        ar = max(w, h) / min(w, h) if min(w, h) > 1e-9 else 999

        # Distance to nearest die edge
        dist_to_edge = min(
            minx - die_x0,
            die_x1 - maxx,
            miny - die_y0,
            die_y1 - maxy,
        )

        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        areas.append(area)
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(ar)
        edge_distances.append(dist_to_edge)
        center_xs.append(cx)
        center_ys.append(cy)
        is_rect.append(rect_ratio > 0.9 and len(pts) <= 6)

    if not areas:
        return None

    areas = np.array(areas)
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    edge_distances = np.array(edge_distances)
    is_rect = np.array(is_rect)

    # What fraction of polygons are near the die perimeter (within 15%)?
    edge_band = 0.15 * min(die_w, die_h)
    near_edge_frac = np.mean(edge_distances < edge_band)

    # Size uniformity: if most polygons are ~same size, std/mean is low
    area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 999

    total_area = np.sum(areas) * (total_count / len(areas)) if len(areas) > 0 else 0

    return {
        "total_count": total_count,
        "sample_size": len(areas),
        "total_area_est": float(total_area),
        "area_pct_of_die": float(total_area / die_area * 100) if die_area > 0 else 0,
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "area_cv": float(area_cv),
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "median_aspect_ratio": float(np.median(aspect_ratios)),
        "pct_rectangular": float(np.mean(is_rect) * 100),
        "pct_near_edge": float(near_edge_frac * 100),
        "mean_edge_distance": float(np.mean(edge_distances)),
    }


# ── Classification rules ────────────────────────────────────────────────────

def classify_layer(stats, die_area):
    """
    Classify a layer based on its geometry statistics.
    Returns (category, confidence, reasoning).
    """
    if stats is None:
        return "unknown", 0, "no polygons"

    count = stats["total_count"]
    mean_a = stats["mean_area"]
    area_cv = stats["area_cv"]
    pct_rect = stats["pct_rectangular"]
    pct_edge = stats["pct_near_edge"]
    ar = stats["median_aspect_ratio"]
    area_pct = stats["area_pct_of_die"]
    max_a = stats["max_area"]

    # Die boundary: 1 polygon covering ~100% of die
    if count <= 2 and area_pct > 80:
        return "die_boundary", 95, f"1-2 polys covering {area_pct:.0f}% of die"

    # Fill: millions of tiny uniform polygons
    if count > 100000 and area_cv < 0.5 and mean_a < 50:
        return "fill", 90, f"{count/1e6:.1f}M uniform tiny polys (CV={area_cv:.2f})"

    # Fill variant: very high count, small area ratio
    if count > 1000000 and area_pct < 10:
        return "fill", 80, f"{count/1e6:.1f}M polys, only {area_pct:.1f}% of die"

    # Pad/bond openings: moderate count, large uniform rectangles near edges
    if (20 < count < 2000
        and mean_a > 5000
        and pct_rect > 60
        and pct_edge > 40
        and area_cv < 1.5):
        conf = 70
        reasons = []
        if pct_edge > 60:
            conf += 10
            reasons.append(f"{pct_edge:.0f}% near edge")
        if pct_rect > 80:
            conf += 5
            reasons.append(f"{pct_rect:.0f}% rectangular")
        if area_cv < 0.8:
            conf += 5
            reasons.append(f"uniform size (CV={area_cv:.2f})")
        return "pad_candidate", min(conf, 95), "; ".join(reasons) or "large rects near edges"

    # Top metal / RDL: few large polygons, high die coverage
    if count < 3000 and mean_a > 10000 and area_pct > 50:
        return "top_metal_or_rdl", 65, f"{count} polys, {area_pct:.0f}% die coverage"

    # Passivation opening: similar to pads but may have different count
    if (count < 5000
        and mean_a > 1000
        and pct_rect > 50
        and pct_edge > 30
        and area_pct < 30):
        return "passivation_or_pad", 50, f"medium rects, {pct_edge:.0f}% near edge"

    # Metal routing: moderate-high polygon count, moderate area
    if 1000 < count < 1000000 and 10 < mean_a < 50000:
        return "metal_routing", 50, f"{count} polys, mean area {mean_a:.0f}"

    # Implant/well: moderate count, large area coverage
    if count < 50000 and area_pct > 20:
        return "implant_or_well", 45, f"{count} polys, {area_pct:.0f}% die coverage"

    # Poly: thin geometries (high aspect ratio)
    if 1000 < count < 500000 and ar > 3 and mean_a < 5000:
        return "poly_candidate", 50, f"thin shapes (AR={ar:.1f}), mean area {mean_a:.0f}"

    # Via/contact: many small uniform polygons but fewer than fill
    if 10000 < count < 1000000 and mean_a < 100 and area_cv < 1.0:
        return "via_or_contact", 55, f"{count} small uniform polys"

    # Text/marker: very few polygons, small area
    if count < 20 and area_pct < 0.1:
        return "marker_or_text", 40, f"only {count} polys, negligible area"

    # Seal ring: few polygons near edge with large total length
    if count < 50 and pct_edge > 80 and area_pct > 1:
        return "seal_ring", 55, f"{count} polys, {pct_edge:.0f}% at edge"

    return "unclassified", 20, f"{count} polys, mean area {mean_a:.1f}, {area_pct:.1f}% die"


# ── Rendering ────────────────────────────────────────────────────────────────

def render_layer_png(cell, layer, datatype, die_bb, output_path, max_render=5000):
    """Render a layer's polygons as a PNG thumbnail."""
    polys, total = sample_polygons(cell, layer, datatype, max_sample=max_render)
    if not polys:
        return

    (die_x0, die_y0), (die_x1, die_y1) = die_bb
    die_w = die_x1 - die_x0
    die_h = die_y1 - die_y0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * die_h / max(die_w, 1)))
    patches = []
    for poly in polys:
        pts = poly.points if isinstance(poly, gdstk.Polygon) else poly
        patches.append(mpatches.Polygon(pts, closed=True))

    if patches:
        pc = PatchCollection(patches, alpha=0.5, facecolor="steelblue",
                             edgecolor="navy", linewidth=0.2)
        ax.add_collection(pc)

    # Draw die outline
    ax.plot([die_x0, die_x1, die_x1, die_x0, die_x0],
            [die_y0, die_y0, die_y1, die_y1, die_y0],
            'r-', linewidth=0.5, alpha=0.5)

    ax.set_xlim(die_x0 - die_w * 0.02, die_x1 + die_w * 0.02)
    ax.set_ylim(die_y0 - die_h * 0.02, die_y1 + die_h * 0.02)
    ax.set_aspect("equal")
    shown = len(polys)
    title = f"Layer {layer}/{datatype}  ({shown}"
    if total > shown:
        title += f" of {total}"
    title += " polys)"
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=7)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-identify GDS layers by analyzing polygon geometry patterns."
    )
    parser.add_argument("gds_file", help="Path to the .gds file")
    parser.add_argument("--output", required=True, help="Output directory for report + PNGs")
    parser.add_argument("--sample", type=int, default=500,
                        help="Max polygons to sample per layer (default 500)")
    parser.add_argument("--render-top", type=int, default=20,
                        help="Render PNGs for the top N most interesting layers (default 20)")
    parser.add_argument("--cell", default=None, help="Target cell (default: top-level)")
    args = parser.parse_args()

    lib, cell, die_bb, die_area = load_gds(args.gds_file)
    if args.cell:
        for c in lib.cells:
            if c.name == args.cell:
                cell = c
                break

    os.makedirs(args.output, exist_ok=True)
    png_dir = os.path.join(args.output, "layer_pngs")
    os.makedirs(png_dir, exist_ok=True)

    # Discover all layer/datatype pairs by scanning cells
    print("Discovering layers ...")
    all_layers = set()
    for c in lib.cells:
        for poly in c.polygons:
            all_layers.add((poly.layer, poly.datatype))
        for path in c.paths:
            for poly in path.to_polygons():
                all_layers.add((poly.layer, poly.datatype))
    all_layers = sorted(all_layers)
    print(f"  Found {len(all_layers)} layer/datatype pairs")

    # Analyze each layer
    results = []
    for i, (layer, dt) in enumerate(all_layers):
        print(f"  [{i+1}/{len(all_layers)}] Analyzing layer {layer}/{dt} ...", end=" ")
        t0 = time.time()
        sample, total = sample_polygons(cell, layer, dt, max_sample=args.sample)
        stats = analyze_polygons(sample, total, die_bb, die_area)
        category, confidence, reasoning = classify_layer(stats, die_area)
        elapsed = time.time() - t0
        print(f"{category} ({confidence}%) — {elapsed:.1f}s")

        entry = {
            "layer": layer,
            "datatype": dt,
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        if stats:
            entry["stats"] = stats
        results.append(entry)

    # Sort: pad candidates and high-confidence first
    priority_order = [
        "pad_candidate", "passivation_or_pad", "top_metal_or_rdl",
        "seal_ring", "die_boundary", "poly_candidate", "via_or_contact",
        "metal_routing", "implant_or_well", "fill", "marker_or_text",
        "unclassified", "unknown"
    ]
    priority_map = {cat: i for i, cat in enumerate(priority_order)}

    results.sort(key=lambda r: (priority_map.get(r["category"], 99), -r["confidence"]))

    # Render PNGs for the most interesting layers
    render_candidates = [r for r in results if r["category"] not in ("fill", "unknown")]
    render_candidates = render_candidates[:args.render_top]

    print(f"\nRendering {len(render_candidates)} layer PNGs ...")
    for r in render_candidates:
        layer, dt = r["layer"], r["datatype"]
        png_path = os.path.join(png_dir, f"layer_{layer}_{dt}.png")
        print(f"  Rendering layer {layer}/{dt} ({r['category']}) ...", end=" ")
        t0 = time.time()
        render_layer_png(cell, layer, dt, die_bb, png_path)
        print(f"{time.time() - t0:.1f}s")
        r["png"] = f"layer_pngs/layer_{layer}_{dt}.png"

    # Write JSON report
    json_path = os.path.join(args.output, "layer_identification.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Write human-readable summary
    summary_path = os.path.join(args.output, "layer_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== GDS LAYER AUTO-IDENTIFICATION ===\n")
        f.write(f"File: {args.gds_file}\n")
        f.write(f"Cell: {cell.name}\n")
        f.write(f"Die area: {die_area:.0f}\n")
        f.write(f"Total layers: {len(results)}\n\n")

        # Group by category
        by_cat = defaultdict(list)
        for r in results:
            by_cat[r["category"]].append(r)

        for cat in priority_order:
            if cat not in by_cat:
                continue
            layers = by_cat[cat]
            f.write(f"\n--- {cat.upper().replace('_', ' ')} ---\n")
            for r in layers:
                s = r.get("stats", {})
                f.write(f"  Layer {r['layer']}/{r['datatype']}  "
                        f"conf={r['confidence']}%  "
                        f"polys={s.get('total_count', '?')}  "
                        f"area={s.get('total_area_est', 0):.0f}  "
                        f"({r['reasoning']})\n")

        # Auto-generated layer_config suggestion
        pad_layers = [r for r in results if r["category"] in ("pad_candidate", "passivation_or_pad")]
        metal_layers = [r for r in results if r["category"] in ("top_metal_or_rdl", "metal_routing")]

        f.write("\n\n=== SUGGESTED layer_config.json ===\n")
        f.write("(Review the PNGs to confirm before using!)\n\n")

        config_suggestion = {
            "pad_layers": [[r["layer"], r["datatype"]] for r in pad_layers[:3]],
            "rdl_layers": [[r["layer"], r["datatype"]]
                           for r in results
                           if r["category"] == "top_metal_or_rdl"][:2],
            "check_layers": [],
            "min_overlap_area": 1.0,
        }

        # Add interesting check layers
        for r in results:
            if r["category"] in ("poly_candidate",):
                config_suggestion["check_layers"].append(
                    {"layer": [r["layer"], r["datatype"]], "name": f"poly_{r['layer']}"})
            elif r["category"] == "metal_routing":
                config_suggestion["check_layers"].append(
                    {"layer": [r["layer"], r["datatype"]], "name": f"metal_{r['layer']}"})
            elif r["category"] == "implant_or_well":
                config_suggestion["check_layers"].append(
                    {"layer": [r["layer"], r["datatype"]], "name": f"implant_{r['layer']}"})

        # Limit check layers to a reasonable number
        config_suggestion["check_layers"] = config_suggestion["check_layers"][:10]

        f.write(json.dumps(config_suggestion, indent=2))
        f.write("\n")

    # Also write the suggested config as its own file
    config_path = os.path.join(args.output, "suggested_layer_config.json")
    with open(config_path, "w") as f:
        json.dump(config_suggestion, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"  Summary:        {summary_path}")
    print(f"  Full JSON:      {json_path}")
    print(f"  Suggested config: {config_path}")
    print(f"  Layer PNGs:     {png_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Open the PNGs in {png_dir}/ to visually confirm layer IDs")
    print(f"  2. Edit {config_path} if needed")
    print(f"  3. Run: python find_overlaps.py {args.gds_file} --config {config_path}")


if __name__ == "__main__":
    main()
