#!/usr/bin/env python3
"""
GDS Layout Risk Analysis — End-to-End Pipeline

One command to go from GDS file to AI risk assessment:
  1. Auto-identify layers (pads, poly, metal, etc.)
  2. Detect overlaps under pad regions
  3. Render visual overlays for concerning pads
  4. Send findings + images to LLM for risk assessment

Usage:
    python run_pipeline.py path/to/chip.gds --output results/
    python run_pipeline.py path/to/chip.gds --output results/ --llm-url http://localhost:8000
    python run_pipeline.py path/to/chip.gds --output results/ --no-llm   (skip LLM, just generate data)
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import defaultdict

import gdstk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

import urllib.request
import urllib.error


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Layer identification
# ═══════════════════════════════════════════════════════════════════════════════

def identify_layers(lib, cell, die_bb, die_area, sample_size=500):
    """
    Auto-classify layers by analyzing polygon geometry patterns.
    Returns a dict with pad_layers, check_layers, etc.
    """
    print("\n[Step 1] Identifying layers ...")

    all_layers = set()
    for c in lib.cells:
        for poly in c.polygons:
            all_layers.add((poly.layer, poly.datatype))
        for path in c.paths:
            for poly in path.to_polygons():
                all_layers.add((poly.layer, poly.datatype))
    all_layers = sorted(all_layers)
    print(f"  Found {len(all_layers)} layer/datatype pairs")

    (die_x0, die_y0), (die_x1, die_y1) = die_bb
    die_w = die_x1 - die_x0
    die_h = die_y1 - die_y0

    classifications = {}

    for i, (layer, dt) in enumerate(all_layers):
        if (i + 1) % 10 == 0:
            print(f"  Analyzing layer {i+1}/{len(all_layers)} ...", end="\r")

        raw_polys = cell.get_polygons(layer=layer, datatype=dt)
        total = len(raw_polys)
        if total == 0:
            continue

        # Sample
        if total <= sample_size:
            sample = raw_polys
        else:
            indices = np.linspace(0, total - 1, sample_size, dtype=int)
            sample = [raw_polys[i] for i in indices]

        areas = []
        near_edge = 0
        rect_count = 0
        aspect_ratios = []

        for poly in sample:
            pts = poly.points if isinstance(poly, gdstk.Polygon) else poly
            if len(pts) < 3:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            w = xs.max() - xs.min()
            h = ys.max() - ys.min()
            area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
            if area < 1e-12:
                continue

            areas.append(area)
            bbox_area = w * h if w > 0 and h > 0 else area
            if bbox_area > 0 and area / bbox_area > 0.9 and len(pts) <= 6:
                rect_count += 1

            ar = max(w, h) / min(w, h) if min(w, h) > 1e-9 else 999
            aspect_ratios.append(ar)

            dist = min(xs.min() - die_x0, die_x1 - xs.max(),
                       ys.min() - die_y0, die_y1 - ys.max())
            if dist < 0.15 * min(die_w, die_h):
                near_edge += 1

        if not areas:
            continue

        areas = np.array(areas)
        mean_a = np.mean(areas)
        area_cv = np.std(areas) / mean_a if mean_a > 0 else 999
        total_area_est = np.sum(areas) * (total / len(areas))
        area_pct = total_area_est / die_area * 100 if die_area > 0 else 0
        pct_rect = rect_count / len(areas) * 100
        pct_edge = near_edge / len(areas) * 100
        med_ar = np.median(aspect_ratios) if aspect_ratios else 1

        # Classification logic
        cat = "unclassified"
        conf = 20

        if total <= 2 and area_pct > 80:
            cat, conf = "die_boundary", 95
        elif total > 1000000 and area_pct < 10:
            cat, conf = "fill", 90
        elif total > 100000 and area_cv < 0.5 and mean_a < 50:
            cat, conf = "fill", 90
        elif (20 < total < 2000 and mean_a > 5000 and pct_rect > 60
              and pct_edge > 40 and area_cv < 1.5):
            cat, conf = "pad", 85
        elif total < 3000 and mean_a > 10000 and area_pct > 50:
            cat, conf = "top_metal_or_rdl", 65
        elif 1000 < total < 500000 and med_ar > 3 and mean_a < 5000:
            cat, conf = "poly", 55
        elif 10000 < total < 1000000 and mean_a < 100 and area_cv < 1.0:
            cat, conf = "via_or_contact", 55
        elif 1000 < total < 1000000 and 10 < mean_a < 50000:
            cat, conf = "metal_routing", 50
        elif total < 50000 and area_pct > 20:
            cat, conf = "implant_or_well", 45
        elif total < 20 and area_pct < 0.1:
            cat, conf = "marker", 40

        classifications[(layer, dt)] = {
            "category": cat,
            "confidence": conf,
            "total_count": total,
            "mean_area": float(mean_a),
            "area_pct": float(area_pct),
            "pct_rect": float(pct_rect),
            "pct_edge": float(pct_edge),
        }

    print()

    # Build config from classifications
    pad_layers = []
    poly_layers = []
    metal_layers = []
    rdl_layers = []

    for (layer, dt), info in sorted(classifications.items()):
        cat = info["category"]
        if cat == "pad":
            pad_layers.append([layer, dt])
        elif cat == "poly":
            poly_layers.append({"layer": [layer, dt], "name": f"poly_{layer}"})
        elif cat == "metal_routing":
            metal_layers.append({"layer": [layer, dt], "name": f"metal_{layer}"})
        elif cat == "top_metal_or_rdl":
            rdl_layers.append([layer, dt])

    # Only use top 5 metal layers by polygon count (most important)
    metal_layers.sort(key=lambda x: classifications[tuple(x["layer"])]["total_count"], reverse=True)
    metal_layers = metal_layers[:5]

    config = {
        "pad_layers": pad_layers,
        "rdl_layers": [],  # skip RDL to avoid noise
        "check_layers": poly_layers + metal_layers,
        "min_overlap_area": 10.0,
    }

    print(f"  Identified: {len(pad_layers)} pad layers, {len(poly_layers)} poly layers, "
          f"{len(metal_layers)} metal layers")
    if pad_layers:
        print(f"  Pad layers: {pad_layers}")

    return config, classifications


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Overlap detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_overlaps(lib, cell, config):
    """Detect what geometries sit under pad regions."""
    print("\n[Step 2] Detecting overlaps ...")

    layers_of_interest = set()
    for ldt in config["pad_layers"]:
        layers_of_interest.add(tuple(ldt))
    for cl in config["check_layers"]:
        layers_of_interest.add(tuple(cl["layer"]))

    # Extract polygons per layer
    layer_polys = {}
    for ldt in layers_of_interest:
        layer, datatype = ldt
        print(f"  Extracting layer {layer}/{datatype} ...", end=" ")
        raw = cell.get_polygons(layer=layer, datatype=datatype)
        polys = []
        for gds_poly in raw:
            pts = gds_poly.points if isinstance(gds_poly, gdstk.Polygon) else gds_poly
            if len(pts) < 3:
                continue
            try:
                sp = ShapelyPolygon(pts)
                if not sp.is_valid:
                    sp = make_valid(sp)
                if not sp.is_empty and sp.area > 0:
                    polys.append(sp)
            except Exception:
                pass
        print(f"{len(polys)} polygons")
        if polys:
            layer_polys[ldt] = polys

    # Gather pad polygons
    pad_polys = []
    for ldt in config["pad_layers"]:
        pad_polys.extend(layer_polys.get(tuple(ldt), []))
    print(f"  Total pad regions: {len(pad_polys)}")

    # Filter out tiny polygons (likely not real pads)
    if pad_polys:
        areas = [p.area for p in pad_polys]
        median_area = np.median(areas)
        # Real pads should be at least 10% of median pad size
        min_pad_area = median_area * 0.1
        real_pads = [p for p in pad_polys if p.area >= min_pad_area]
        filtered = len(pad_polys) - len(real_pads)
        if filtered > 0:
            print(f"  Filtered {filtered} tiny polygons (area < {min_pad_area:.1f}), "
                  f"keeping {len(real_pads)} real pads")
        pad_polys = real_pads

    # Build spatial index per check layer
    min_area = config["min_overlap_area"]
    check_indexes = {}
    for cl in config["check_layers"]:
        ldt = tuple(cl["layer"])
        polys = layer_polys.get(ldt, [])
        if polys:
            tree = STRtree(polys)
            check_indexes[cl["name"]] = (ldt, polys, tree)

    # Find overlaps
    findings = []
    for idx, pad in enumerate(pad_polys, 1):
        if idx % 10 == 0 or idx == len(pad_polys):
            print(f"  Checking pad {idx}/{len(pad_polys)} ...", end="\r")

        pad_bb = pad.bounds
        pad_area = pad.area

        for layer_name, (ldt, polys, tree) in check_indexes.items():
            candidates = tree.query(pad)
            for ci in candidates:
                geom = polys[int(ci)]
                if not pad.intersects(geom):
                    continue
                try:
                    intersection = pad.intersection(geom)
                except Exception:
                    continue
                overlap_area = intersection.area
                if overlap_area < min_area:
                    continue

                findings.append({
                    "region_type": "pad",
                    "region_bounds": list(pad_bb),
                    "region_area": round(pad_area, 4),
                    "check_layer": layer_name,
                    "check_layer_number": list(ldt),
                    "overlap_area": round(overlap_area, 4),
                    "overlap_pct_of_region": round(100.0 * overlap_area / pad_area, 2) if pad_area > 0 else 0,
                    "overlap_pct_of_geometry": round(100.0 * overlap_area / geom.area, 2) if geom.area > 0 else 0,
                    "geometry_bounds": list(geom.bounds),
                    "geometry_area": round(geom.area, 4),
                })

    print(f"\n  Found {len(findings)} overlaps")
    return findings, pad_polys


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Group findings and score risk
# ═══════════════════════════════════════════════════════════════════════════════

def group_and_score(findings):
    """Group findings by pad, score risk, return sorted list."""
    regions = defaultdict(lambda: {
        "type": None, "bounds": None, "area": 0,
        "layers": defaultdict(lambda: {"count": 0, "total_overlap_area": 0,
                                       "max_overlap_pct": 0})
    })

    for f in findings:
        key = tuple(f["region_bounds"])
        r = regions[key]
        r["type"] = f["region_type"]
        r["bounds"] = f["region_bounds"]
        r["area"] = f["region_area"]
        layer = f["check_layer"]
        r["layers"][layer]["count"] += 1
        r["layers"][layer]["total_overlap_area"] += f["overlap_area"]
        r["layers"][layer]["max_overlap_pct"] = max(
            r["layers"][layer]["max_overlap_pct"], f["overlap_pct_of_region"])

    scored = []
    for key, region in regions.items():
        score = 0
        reasons = []
        for layer, stats in region["layers"].items():
            lname = layer.lower()
            if "poly" in lname:
                score += 40
                reasons.append(f"POLY under pad: {stats['count']} structures, "
                               f"{stats['total_overlap_area']:.1f} area")
            elif "active" in lname or "diff" in lname:
                score += 30
                reasons.append(f"ACTIVE under pad: {stats['count']} structures")
            elif "metal" in lname and stats["max_overlap_pct"] > 50:
                score += 10
                reasons.append(f"{layer}: large overlap ({stats['max_overlap_pct']:.0f}% of pad)")

        if score >= 30:
            level = "HIGH"
        elif score >= 15:
            level = "MEDIUM"
        elif score > 0:
            level = "LOW"
        else:
            level = "CLEAR"

        scored.append({
            "key": key,
            "region": region,
            "level": level,
            "score": score,
            "reasons": reasons,
        })

    scored.sort(key=lambda x: -x["score"])
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Render visual overlays for concerning pads
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_COLORS = {
    "poly": "#FF0000",      # red — dangerous
    "active": "#FF6600",    # orange — dangerous
    "metal": "#4488CC",     # blue — normal routing
}


def get_layer_color(layer_name):
    for key, color in LAYER_COLORS.items():
        if key in layer_name.lower():
            return color
    return "#888888"


def render_pad_overlay(cell, pad_bounds, pad_area, layers_under, findings_for_pad,
                       config, output_path, pad_label=""):
    """
    Render a zoomed-in view of a pad showing all underlying structures,
    color-coded by layer type.
    """
    b = pad_bounds
    pad_cx = (b[0] + b[2]) / 2
    pad_cy = (b[1] + b[3]) / 2
    pad_w = b[2] - b[0]
    pad_h = b[3] - b[1]

    # Zoom out 3x around the pad for context
    margin = max(pad_w, pad_h) * 1.5
    view_x0 = pad_cx - margin
    view_x1 = pad_cx + margin
    view_y0 = pad_cy - margin
    view_y1 = pad_cy + margin

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw pad region (yellow highlight)
    pad_rect = mpatches.FancyBboxPatch(
        (b[0], b[1]), pad_w, pad_h,
        boxstyle="round,pad=0", facecolor="#FFFF00", alpha=0.3,
        edgecolor="#FF8800", linewidth=2)
    ax.add_patch(pad_rect)

    # Draw overlapping geometries
    legend_entries = {}
    for f in findings_for_pad:
        layer_name = f["check_layer"]
        gb = f["geometry_bounds"]
        color = get_layer_color(layer_name)

        # Get actual polygons from the cell for this layer in the view area
        ldt = tuple(f["check_layer_number"])
        gw = gb[2] - gb[0]
        gh = gb[3] - gb[1]
        rect = mpatches.FancyBboxPatch(
            (gb[0], gb[1]), gw, gh,
            boxstyle="round,pad=0", facecolor=color, alpha=0.5,
            edgecolor=color, linewidth=1)
        ax.add_patch(rect)

        if layer_name not in legend_entries:
            legend_entries[layer_name] = mpatches.Patch(
                facecolor=color, alpha=0.5, label=layer_name)

    # Pad outline label
    legend_entries["PAD"] = mpatches.Patch(
        facecolor="#FFFF00", alpha=0.3, edgecolor="#FF8800", label="PAD region")

    ax.set_xlim(view_x0, view_x1)
    ax.set_ylim(view_y0, view_y1)
    ax.set_aspect("equal")
    ax.legend(handles=list(legend_entries.values()), loc="upper right", fontsize=9)
    ax.set_title(f"{pad_label}\nPad at ({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f})  "
                 f"area={pad_area:.0f}", fontsize=11)
    ax.tick_params(labelsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: LLM analysis
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(scored_pads, gds_name):
    """Build a well-calibrated prompt that won't over-react to normal routing."""
    concerning = [s for s in scored_pads if s["level"] in ("HIGH", "MEDIUM")]

    prompt = f"""You are an IC packaging and layout reliability engineer reviewing overlap findings from the GDSII layout of **{gds_name}**, an MPS (Monolithic Power Systems) DC-DC converter IC.

## Important Context — Read Before Assessing

- **Metal routing under pads is NORMAL** in modern IC processes, especially power ICs. Lower metals (metal1-metal4) are routinely routed under pads. This is NOT automatically a defect.
- **What IS concerning:** Poly resistors, active devices (transistors), thin-film resistors, or sensitive analog structures under pads. These can shift during solder reflow (~1-5%) or crack from bonding stress.
- **Pad Keep-Out Zones (KOZ)** rules vary by foundry process. Some processes have strict KOZ for poly only, others for all active layers. Without knowing the specific DRC rules, flag poly/active under pads as WORTH INVESTIGATING, not as definite failures.
- **Risk levels should be:**
  - HIGH: Poly or active device directly under a pad with significant area overlap — needs design team review
  - MEDIUM: Dense metal routing that could cause stress issues, or poly near (but not under) a pad
  - LOW: Normal metal routing under pads — expected and acceptable
  - CLEAR: No concerning structures

## Findings

Total pads analyzed: {len(scored_pads)}
Concerning pads (HIGH/MEDIUM): {len(concerning)}

"""
    if not concerning:
        prompt += "No HIGH or MEDIUM risk pads found. The layout appears clean.\n"
    else:
        for i, s in enumerate(concerning, 1):
            b = s["region"]["bounds"]
            prompt += f"""### Pad {i} at ({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f}) — pre-scored as {s['level']}
Pad area: {s['region']['area']:.1f}
Structures under this pad:
"""
            for layer, stats in sorted(s["region"]["layers"].items()):
                prompt += (f"  - {layer}: {stats['count']} geometries, "
                           f"total overlap {stats['total_overlap_area']:.1f}, "
                           f"max overlap {stats['max_overlap_pct']:.1f}% of pad\n")
            prompt += "\n"

    prompt += """## Your Task

1. For each pad, provide your **risk assessment** (HIGH / MEDIUM / LOW / ACCEPTABLE)
2. Explain your reasoning — distinguish between genuinely concerning overlaps (poly, active) and normal metal routing
3. For HIGH risk pads, recommend specific actions
4. Provide a brief **overall summary** (2-3 sentences) of the layout's packaging health

Be calibrated — metal under pads is normal. Only flag things that could actually cause reliability failures.
"""
    return prompt


def build_vision_message(prompt_text, image_paths):
    """Build an OpenAI-format message with text + images."""
    content = []

    # Add images first
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    # Add text
    content.append({"type": "text", "text": prompt_text})

    return content


def query_llm(messages, url, model, max_tokens=8192):
    """Send messages to the LLM server."""
    endpoint = f"{url}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        # Handle different response formats from vLLM/SGLang
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            content = None

        # SGLang with reasoning model may put response in 'reasoning_content'
        if content is None:
            try:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            except (KeyError, IndexError):
                pass

        if content is None:
            # Dump full response for debugging
            content = f"(Could not parse LLM response. Raw: {json.dumps(data, indent=2)[:2000]})"

        return content


def detect_model(url):
    """Auto-detect which model is loaded on the server."""
    try:
        req = urllib.request.Request(f"{url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GDS layout risk analysis pipeline."
    )
    parser.add_argument("gds_file", help="Path to the .gds file")
    parser.add_argument("--output", required=True, help="Output directory for all results")
    parser.add_argument("--llm-url", default="http://localhost:8000",
                        help="vLLM/SGLang server URL (default: http://localhost:8000)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM analysis (just generate data + images)")
    parser.add_argument("--use-vision", action="store_true",
                        help="Send pad overlay images to LLM (requires multimodal model)")
    parser.add_argument("--cell", default=None, help="Target cell (default: top-level)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t_start = time.time()

    # ── Load GDS ──────────────────────────────────────────────────────────
    print(f"Loading {args.gds_file} ...")
    t0 = time.time()
    lib = gdstk.read_gds(args.gds_file)
    top = lib.top_level()
    if not top:
        print("Error: no top-level cells.")
        sys.exit(1)
    cell = top[0] if not args.cell else None
    if args.cell:
        for c in lib.cells:
            if c.name == args.cell:
                cell = c
                break
    if cell is None:
        print(f"Error: cell '{args.cell}' not found.")
        sys.exit(1)
    die_bb = cell.bounding_box()
    (x0, y0), (x1, y1) = die_bb
    die_area = (x1 - x0) * (y1 - y0)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(lib.cells)} cells, "
          f"cell={cell.name}, die={die_area:.0f}")

    # ── Step 1: Identify layers ──────────────────────────────────────────
    config, classifications = identify_layers(lib, cell, die_bb, die_area)

    config_path = os.path.join(args.output, "auto_layer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    if not config["pad_layers"]:
        print("\nError: no pad layers identified. Cannot continue.")
        print("Check the layer PNGs and create a manual layer_config.json")
        sys.exit(1)

    # ── Step 2: Detect overlaps ──────────────────────────────────────────
    findings, pad_polys = detect_overlaps(lib, cell, config)

    findings_path = os.path.join(args.output, "overlap_findings.json")
    result = {
        "gds_file": os.path.basename(args.gds_file),
        "cell": cell.name,
        "config": config,
        "summary": {
            "total_pad_regions": len(pad_polys),
            "total_rdl_regions": 0,
            "total_overlaps": len(findings),
        },
        "findings": findings,
    }
    with open(findings_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Findings saved to {findings_path}")

    # ── Step 3: Score and group ──────────────────────────────────────────
    print("\n[Step 3] Scoring risk ...")
    scored = group_and_score(findings)
    risk_counts = defaultdict(int)
    for s in scored:
        risk_counts[s["level"]] += 1
    print(f"  HIGH: {risk_counts['HIGH']}  MEDIUM: {risk_counts['MEDIUM']}  "
          f"LOW: {risk_counts['LOW']}  CLEAR: {risk_counts['CLEAR']}")

    # ── Step 4: Render overlays for HIGH/MEDIUM pads ─────────────────────
    print("\n[Step 4] Rendering pad overlays ...")
    overlay_dir = os.path.join(args.output, "pad_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    concerning = [s for s in scored if s["level"] in ("HIGH", "MEDIUM")]
    overlay_paths = []

    for i, s in enumerate(concerning[:20], 1):  # max 20 overlays
        b = s["region"]["bounds"]
        # Get findings for this specific pad
        pad_findings = [f for f in findings if tuple(f["region_bounds"]) == s["key"]]

        png_path = os.path.join(overlay_dir, f"pad_{i}_{s['level'].lower()}.png")
        render_pad_overlay(
            cell, b, s["region"]["area"], s["region"]["layers"],
            pad_findings, config, png_path,
            pad_label=f"Pad {i} — {s['level']} RISK"
        )
        overlay_paths.append(png_path)
        print(f"  Rendered pad {i}/{len(concerning)} ({s['level']})")

    # ── Step 5: LLM analysis ─────────────────────────────────────────────
    if args.no_llm:
        print("\n[Step 5] Skipping LLM analysis (--no-llm)")
    else:
        print("\n[Step 5] Running LLM risk assessment ...")
        model = detect_model(args.llm_url)
        if not model:
            print(f"  Warning: could not connect to LLM at {args.llm_url}")
            print("  Run with --no-llm to skip, or start your LLM server")
            args.no_llm = True
        else:
            print(f"  Model: {model}")
            prompt_text = build_llm_prompt(scored, os.path.basename(args.gds_file))

            # Save prompt for reference
            prompt_path = os.path.join(args.output, "llm_prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(prompt_text)

            # Build messages
            if args.use_vision and overlay_paths:
                print(f"  Sending {len(overlay_paths)} images + text to model ...")
                # Limit to top 5 images to avoid overwhelming the context
                imgs = overlay_paths[:5]
                content = build_vision_message(prompt_text, imgs)
                messages = [{"role": "user", "content": content}]
            else:
                messages = [
                    {"role": "system",
                     "content": "You are an IC packaging reliability expert. "
                                "Be calibrated — metal under pads is normal in power ICs."},
                    {"role": "user", "content": prompt_text}
                ]

            try:
                t0 = time.time()
                response = query_llm(messages, args.llm_url, model)
                print(f"  LLM responded in {time.time() - t0:.1f}s")

                analysis_path = os.path.join(args.output, "analysis.md")
                with open(analysis_path, "w") as f:
                    f.write(f"# LLM Risk Assessment\n\n")
                    f.write(f"**Model:** {model}\n\n")
                    f.write(response)
                print(f"  Analysis saved to {analysis_path}")
            except Exception as e:
                print(f"  LLM error: {e}")
                args.no_llm = True

    # ── Summary report ────────────────────────────────────────────────────
    report_path = os.path.join(args.output, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# GDS Layout Risk Report — {os.path.basename(args.gds_file)}\n\n")
        f.write(f"**Cell:** {cell.name}  \n")
        f.write(f"**Die size:** {x1-x0:.0f} x {y1-y0:.0f}  \n")
        f.write(f"**Total pads:** {len(pad_polys)}  \n")
        f.write(f"**Total overlaps:** {len(findings)}  \n\n")

        f.write("## Risk Overview\n\n")
        f.write("| Risk Level | Count |\n|---|---|\n")
        for level in ["HIGH", "MEDIUM", "LOW", "CLEAR"]:
            if risk_counts[level] > 0:
                f.write(f"| {level} | {risk_counts[level]} |\n")
        f.write("\n")

        if concerning:
            f.write("## Concerning Pads\n\n")
            for i, s in enumerate(concerning, 1):
                b = s["region"]["bounds"]
                f.write(f"### Pad {i} at ({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f}) — {s['level']}\n\n")
                f.write(f"Area: {s['region']['area']:.1f}  \n")
                if s["reasons"]:
                    f.write("**Risk factors:**\n")
                    for r in s["reasons"]:
                        f.write(f"- {r}\n")
                f.write("\n| Layer | Count | Overlap Area | Max % |\n|---|---|---|---|\n")
                for layer, stats in sorted(s["region"]["layers"].items()):
                    f.write(f"| {layer} | {stats['count']} | "
                            f"{stats['total_overlap_area']:.1f} | "
                            f"{stats['max_overlap_pct']:.1f}% |\n")
                if i <= len(overlay_paths):
                    rel_path = os.path.relpath(overlay_paths[i-1], args.output)
                    f.write(f"\n![Pad {i} overlay]({rel_path})\n")
                f.write("\n")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Report:    {report_path}")
    print(f"  Overlays:  {overlay_dir}/")
    if not args.no_llm:
        print(f"  Analysis:  {os.path.join(args.output, 'analysis.md')}")
    print(f"  Config:    {config_path}")
    print(f"  Findings:  {findings_path}")


if __name__ == "__main__":
    main()
