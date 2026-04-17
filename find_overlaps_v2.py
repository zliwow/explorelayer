#!/usr/bin/env python3
"""
Hierarchy-aware overlap detector — the v2 version of find_overlaps.py.

What this adds on top of v1:
  - every finding carries the owning cell-instance path from the GDS
    (e.g. 'mpq8897r4_1/XANA/XCMBLK/XLDO_VBIAS/met3_polygon'), so the
    LLM can reason about WHAT the overlap is crossing, not just geometry
  - each finding gets a heuristic severity tier (high/medium/low/info)
    driven by name-pattern matching (BGREF, VBIAS, LDO, MATCH, MIRROR,
    OSC, PLL, ESD, pad, fill, ...)
  - optional CDL cross-reference: if a cell name matches a subckt in
    the netlist, we attach the subckt's pin/device count
  - findings are grouped by owner path AND layer so matched-pair
    devices and repeated blocks collapse into one high-value item

This is meant as the feeder for run_llm_analysis.py — small number of
well-labeled findings instead of 39/39 geometric noise.

Usage:
    python find_overlaps_v2.py chip.gds --config layer_config.json
    python find_overlaps_v2.py chip.gds --config layer_config.json --netlist chip.cdl
    python find_overlaps_v2.py chip.gds --config layer_config.json --output findings_v2.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import gdstk
from shapely.geometry import Point, Polygon as ShapelyPolygon, box as shapely_box
from shapely.strtree import STRtree
from shapely.validation import make_valid


# ── Severity heuristics ────────────────────────────────────────────────

SEVERITY_PATTERNS = [
    # (regex, tier, reason). Order roughly most → least specific.
    # Patterns are matched against the LEAF segment (deepest cell name) of
    # the owner path. That's deliberate — matching against the whole path
    # makes everything inherit "high" from a chip-level wrapper like
    # MPQ8897_ESD_R3, which is the ESD wrapper around the entire die, not
    # a real ESD clamp device.
    (r"(?i)(^|_)(bg|bgref|bandgap|vref|ibias|vbias|iref)(_|<|$)",
     "high",
     "bandgap / voltage or current reference — RDL crossings skew matching and introduce stress drift"),
    (r"(?i)(^|_)(match|mirror|diffpair|diff_pair|otaamp|opamp|preamp|cmr)(_|<|$)",
     "high",
     "matched pair / current mirror / amplifier front-end — critical for offset and CMRR"),
    (r"(?i)(^|_)(esdclamp|esd_clamp|clamp)(_|<|$)",
     "high",
     "ESD clamp device — unexpected overlap may affect discharge path"),
    (r"(?i)(^|_)(ldo|buck|boost|pfm|pwm|regulator)(_|<|$)",
     "medium",
     "regulator / power-conversion block — sensitive to thermal and metal coupling"),
    (r"(?i)(^|_)(adc|dac|sar|comparator)(_|<|$)",
     "medium",
     "converter front-end — INL/DNL sensitive to asymmetry"),
    (r"(?i)(^|_)(osc|pll|pfd|vco)(_|<|$)",
     "medium",
     "oscillator / PLL / clock — jitter sensitive to substrate coupling"),
    (r"(?i)(^|_)(pad|bond|vss|vdd|gnd|agnd|pgnd)(_|<|$)",
     "low",
     "pad / power rail — usually an intentional stackup"),
    (r"(?i)(^|_)(fill|filler|decap|tap|guard|dummy)(_|<|$)",
     "low",
     "fill / decap / guard — typically intentional layout feature"),
]

# Wrapper / IP names that should always be downgraded — they're the chip-level
# wrapper or vendor IP and the polygons inside aren't actually analog-sensitive.
WRAPPER_PATTERNS = [
    (r"(?i)package", "info", "chip-level package wrapper"),
    (r"(?i)esd_r\d+$", "info", "chip-level ESD wrapper, not a real clamp"),
    (r"(?i)top_r\d+$", "info", "chip-level top wrapper"),
    (r"(?i)(mem|sram|rom|ram|cell_flat|x[a-z0-9]*memory|xr[s]?\d+[a-z0-9]+)",
     "low",
     "memory IP block — RDL routing over compiled memory is usually intentional"),
]

TIER_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


def _path_segments(path_str):
    return [s for s in path_str.split("/") if s] if path_str else []


def _is_wrapper(segment):
    """True if this segment is a chip-level wrapper (package/top/ESD wrapper)."""
    return bool(re.search(r"(?i)(package|top_r\d+$|esd_r\d+$)", segment))


def classify_path(path_str):
    """
    Return (tier, reason_list).

    Two-pass logic:
      1. If the LEAF is a wrapper / vendor IP block, that's the ceiling
         (chip-level wrapper or compiled memory should never get 'high'
         even if an ancestor name happens to contain 'esd' or 'match').
      2. Otherwise scan all NON-wrapper segments for severity keywords.
    """
    segs = _path_segments(path_str)
    if not segs:
        return "info", []
    leaf = segs[-1]

    for regex, tier, reason in WRAPPER_PATTERNS:
        if re.search(regex, leaf):
            return tier, [f"[{tier}] {reason} (leaf={leaf})"]

    # Scan severity patterns against any non-wrapper segment.
    matched = []
    candidates = [s for s in segs if not _is_wrapper(s)]
    for seg in candidates:
        for regex, tier, reason in SEVERITY_PATTERNS:
            if re.search(regex, seg):
                matched.append((tier, f"{reason} (segment={seg})"))
    if not matched:
        return "info", []
    best_tier = max((t for t, _ in matched), key=lambda t: TIER_ORDER[t])
    reasons = [f"[{t}] {r}" for t, r in matched]
    return best_tier, reasons


# ── Hierarchy walk — builds an instance bbox index ─────────────────────

def walk_instance_bboxes(top_cell):
    """
    Walk the hierarchy under top_cell, returning a list of
    (shapely_bbox, path_tuple, depth) for every instance.

    Translation-only composition — gdstk rotations/mirrors aren't composed
    here. For owner-name attribution (which is what we're after) the
    resulting bboxes are still accurate enough: we only need the surrounding
    cell context, not pixel-perfect geometry.
    """
    entries = []
    stack = [(top_cell, (0.0, 0.0), (top_cell.name,))]
    seen_paths = set()

    while stack:
        cell, origin, path = stack.pop()
        if path in seen_paths:
            continue
        seen_paths.add(path)

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
                # Guard against pathological recursion
                if len(new_path) > 64:
                    continue
                stack.append((child, new_origin, new_path))

    return entries


def resolve_owner(point_xy, bbox_entries, bbox_tree):
    """
    Find the DEEPEST cell-instance path whose bbox contains point_xy.
    Falls back to None if nothing contains it.
    """
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


# ── CDL subckt catalog (lightweight) ───────────────────────────────────

PARAM_RE = re.compile(r"^\w+\s*=")


def load_subckt_catalog(cdl_path):
    """
    Parse a CDL/SPICE netlist, return {subckt_name_lower: {
        'name': original_name,
        'pin_count': int,
        'device_counts': {'M': n, 'R': n, 'C': n, ...}
    }}.
    Folds '+' continuation lines when counting pins.
    """
    if not cdl_path or not os.path.exists(cdl_path):
        return {}

    catalog = {}
    pending_header = None
    pending_devices = Counter()

    def flush():
        nonlocal pending_header, pending_devices
        if pending_header is None:
            return
        name = pending_header["name"]
        tokens = pending_header["tokens"]
        pins = [t for t in tokens if "=" not in t]
        catalog[name.lower()] = {
            "name": name,
            "pin_count": len(pins),
            "device_counts": dict(pending_devices),
        }
        pending_header = None
        pending_devices = Counter()

    with open(cdl_path, "r", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            first = stripped[0]
            if first in ("*", "$") or stripped.startswith("//"):
                continue
            if first == "+":
                if pending_header is not None:
                    pending_header["tokens"].extend(stripped[1:].strip().split())
                continue
            low = stripped.lower()
            if low.startswith(".subckt") or low.startswith("subckt "):
                flush()
                parts = stripped.split()
                if len(parts) >= 2:
                    pending_header = {"name": parts[1], "tokens": parts[2:]}
                continue
            if low.startswith(".ends") or low == "ends":
                flush()
                continue
            # Device line — count by prefix letter
            if pending_header is not None and first.upper() in "MRCDQJXLKVI":
                pending_devices[first.upper()] += 1

    flush()
    return catalog


# ── Geometry extraction (same shape as v1) ─────────────────────────────

def load_config(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    required = ["pad_layers", "check_layers"]
    for key in required:
        if key not in cfg:
            print(f"Error: config missing required key '{key}'")
            sys.exit(1)
    cfg["pad_layers"] = [tuple(l) for l in cfg["pad_layers"]]
    cfg["rdl_layers"] = [tuple(l) for l in cfg.get("rdl_layers", [])]
    for cl in cfg["check_layers"]:
        cl["layer"] = tuple(cl["layer"])
    cfg.setdefault("min_overlap_area", 0.0)
    cfg.setdefault("sensitivity_ranking", {})
    return cfg


def get_target_cell(lib, cell_name=None):
    if cell_name:
        for c in lib.cells:
            if c.name == cell_name:
                return c
        print(f"Error: cell '{cell_name}' not found.")
        sys.exit(1)
    top = lib.top_level()
    if not top:
        print("Error: no top-level cells found.")
        sys.exit(1)
    if len(top) > 1:
        print(f"Note: multiple top cells; using '{top[0].name}'")
    return top[0]


def extract_polygons_for_layers(cell, layers_of_interest):
    """Fast flattened polygons per (layer, datatype) — uses gdstk's C++ path."""
    layer_polys = defaultdict(list)
    for ldt in layers_of_interest:
        layer, datatype = ldt
        print(f"  layer {layer}/{datatype} ... ", end="", flush=True)
        raw_polys = cell.get_polygons(layer=layer, datatype=datatype)
        kept = 0
        for gp in raw_polys:
            pts = gp.points if isinstance(gp, gdstk.Polygon) else gp
            if len(pts) < 3:
                continue
            sp = ShapelyPolygon(pts)
            if not sp.is_valid:
                sp = make_valid(sp)
            if sp.is_empty or sp.area == 0:
                continue
            layer_polys[ldt].append(sp)
            kept += 1
        print(f"{kept} polys")
    return dict(layer_polys)


# ── Overlap detection with owner attribution ───────────────────────────

def path_to_str(path_tuple):
    return "/".join(path_tuple) if path_tuple else ""


def find_overlaps(layer_polys, cfg, bbox_entries, bbox_tree, subckt_catalog):
    min_area = cfg["min_overlap_area"]

    pad_polys = []
    for ldt in cfg["pad_layers"]:
        pad_polys.extend(layer_polys.get(ldt, []))
    rdl_polys = []
    for ldt in cfg["rdl_layers"]:
        rdl_polys.extend(layer_polys.get(ldt, []))

    regions = [(p, "pad") for p in pad_polys] + [(p, "rdl") for p in rdl_polys]
    if not regions:
        print("Warning: no pad or RDL polygons found.")
        return []

    # Spatial indexes per check layer
    check_indexes = {}
    for cl in cfg["check_layers"]:
        ldt = cl["layer"]
        polys = layer_polys.get(ldt, [])
        if polys:
            check_indexes[cl["name"]] = (ldt, polys, STRtree(polys))

    findings = []
    total_regions = len(regions)

    for idx, (region_poly, region_type) in enumerate(regions, 1):
        if idx % 50 == 0 or idx == total_regions:
            print(f"  region {idx}/{total_regions} ...", end="\r", flush=True)

        region_bb = region_poly.bounds
        region_area = region_poly.area
        region_centroid = region_poly.centroid

        for layer_name, (ldt, polys, tree) in check_indexes.items():
            for ci in tree.query(region_poly):
                geom = polys[int(ci)]
                if not region_poly.intersects(geom):
                    continue
                try:
                    inter = region_poly.intersection(geom)
                except Exception:
                    continue
                overlap_area = inter.area
                if overlap_area < min_area:
                    continue

                # Resolve owner path using the geometry polygon's centroid —
                # that's the thing "underneath" the pad/RDL, which is what
                # we care about semantically.
                owner_centroid = geom.centroid
                owner_path = resolve_owner(
                    (owner_centroid.x, owner_centroid.y),
                    bbox_entries, bbox_tree,
                )
                owner_path_str = path_to_str(owner_path) if owner_path else "(unresolved)"
                owner_cell = owner_path[-1] if owner_path else None

                # Heuristic severity from the path
                cfg_tier = cfg["sensitivity_ranking"].get(layer_name, "info")
                heur_tier, reasons = classify_path(owner_path_str)
                # Take the higher of the two tiers
                final_tier = max(
                    (cfg_tier, heur_tier),
                    key=lambda t: TIER_ORDER.get(t, 0),
                )

                subckt_info = None
                if owner_cell and subckt_catalog:
                    sc = subckt_catalog.get(owner_cell.lower())
                    if sc:
                        subckt_info = sc

                findings.append({
                    "region_type": region_type,
                    "region_bounds": [float(v) for v in region_bb],
                    "region_area": round(region_area, 4),
                    "region_centroid": [float(region_centroid.x), float(region_centroid.y)],
                    "check_layer": layer_name,
                    "check_layer_number": list(ldt),
                    "overlap_area": round(overlap_area, 4),
                    "overlap_pct_of_region": round(100.0 * overlap_area / region_area, 2) if region_area > 0 else 0,
                    "overlap_pct_of_geometry": round(100.0 * overlap_area / geom.area, 2) if geom.area > 0 else 0,
                    "geometry_bounds": [float(v) for v in geom.bounds],
                    "geometry_area": round(geom.area, 4),
                    "owner_path": owner_path_str,
                    "owner_cell": owner_cell,
                    "severity_tier": final_tier,
                    "severity_cfg_tier": cfg_tier,
                    "severity_heuristic_tier": heur_tier,
                    "severity_reasons": reasons,
                    "subckt_match": subckt_info,
                })

    print()  # clear the \r line
    return findings


def group_findings(findings):
    """Group by (owner_path, check_layer) so matched-pair devices collapse."""
    groups = defaultdict(lambda: {
        "count": 0,
        "total_overlap_area": 0.0,
        "total_region_area": 0.0,
        "severity_tier": "info",
        "severity_reasons": set(),
        "check_layer": None,
        "owner_path": None,
        "owner_cell": None,
        "subckt_match": None,
        "sample_finding_indices": [],
    })

    for i, f in enumerate(findings):
        key = (f["owner_path"], f["check_layer"])
        g = groups[key]
        g["count"] += 1
        g["total_overlap_area"] += f["overlap_area"]
        g["total_region_area"] += f["region_area"]
        if TIER_ORDER.get(f["severity_tier"], 0) > TIER_ORDER.get(g["severity_tier"], 0):
            g["severity_tier"] = f["severity_tier"]
        g["severity_reasons"].update(f["severity_reasons"])
        g["check_layer"] = f["check_layer"]
        g["owner_path"] = f["owner_path"]
        g["owner_cell"] = f["owner_cell"]
        g["subckt_match"] = f["subckt_match"]
        if len(g["sample_finding_indices"]) < 3:
            g["sample_finding_indices"].append(i)

    # Flatten and sort
    out = []
    for (owner_path, check_layer), g in groups.items():
        out.append({
            "owner_path": owner_path,
            "check_layer": check_layer,
            "severity_tier": g["severity_tier"],
            "severity_reasons": sorted(g["severity_reasons"]),
            "count": g["count"],
            "total_overlap_area": round(g["total_overlap_area"], 4),
            "total_region_area": round(g["total_region_area"], 4),
            "owner_cell": g["owner_cell"],
            "subckt_match": g["subckt_match"],
            "sample_finding_indices": g["sample_finding_indices"],
        })
    out.sort(key=lambda g: (-TIER_ORDER.get(g["severity_tier"], 0),
                            -g["total_overlap_area"]))
    return out


def main():
    p = argparse.ArgumentParser(description="Hierarchy-aware pad/RDL overlap detector.")
    p.add_argument("gds_file")
    p.add_argument("--config", required=True)
    p.add_argument("--cell", default=None)
    p.add_argument("--netlist", default=None, help="Optional CDL/SPICE to cross-reference cell names")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    print(f"Loading {args.gds_file} ...")
    t0 = time.time()
    lib = gdstk.read_gds(args.gds_file)
    print(f"  {len(lib.cells)} cells in {time.time() - t0:.1f}s")

    cell = get_target_cell(lib, args.cell)
    print(f"Top cell: {cell.name}")

    # Build instance bbox index
    print("Building instance bbox index (hierarchy walk) ...")
    t0 = time.time()
    bbox_entries = walk_instance_bboxes(cell)
    bbox_tree = STRtree([e[0] for e in bbox_entries])
    print(f"  {len(bbox_entries)} instances in {time.time() - t0:.1f}s")

    # Optional CDL catalog
    subckt_catalog = {}
    if args.netlist:
        print(f"Loading CDL catalog from {args.netlist} ...")
        t0 = time.time()
        subckt_catalog = load_subckt_catalog(args.netlist)
        print(f"  {len(subckt_catalog)} subckts in {time.time() - t0:.1f}s")

    # Collect layers of interest
    layers_of_interest = set()
    layers_of_interest.update(cfg["pad_layers"])
    layers_of_interest.update(cfg["rdl_layers"])
    for cl in cfg["check_layers"]:
        layers_of_interest.add(cl["layer"])

    print(f"Extracting polygons for {len(layers_of_interest)} layers ...")
    t0 = time.time()
    layer_polys = extract_polygons_for_layers(cell, layers_of_interest)
    total = sum(len(v) for v in layer_polys.values())
    print(f"  {total} polygons in {time.time() - t0:.1f}s")

    print("Detecting overlaps ...")
    t0 = time.time()
    findings = find_overlaps(layer_polys, cfg, bbox_entries, bbox_tree, subckt_catalog)
    print(f"  {len(findings)} raw findings in {time.time() - t0:.1f}s")

    print("Grouping by owner_path × check_layer ...")
    groups = group_findings(findings)

    # Summary of tiers
    tier_counts = Counter(f["severity_tier"] for f in findings)
    group_tier_counts = Counter(g["severity_tier"] for g in groups)

    result = {
        "gds_file": os.path.basename(args.gds_file),
        "cell": cell.name,
        "netlist": os.path.basename(args.netlist) if args.netlist else None,
        "config": {
            "pad_layers": [list(l) for l in cfg["pad_layers"]],
            "rdl_layers": [list(l) for l in cfg["rdl_layers"]],
            "check_layers": [{"name": c["name"], "layer": list(c["layer"])} for c in cfg["check_layers"]],
            "min_overlap_area": cfg["min_overlap_area"],
        },
        "summary": {
            "total_pad_regions": sum(len(layer_polys.get(tuple(l), [])) for l in cfg["pad_layers"]),
            "total_rdl_regions": sum(len(layer_polys.get(tuple(l), [])) for l in cfg["rdl_layers"]),
            "total_raw_findings": len(findings),
            "total_groups": len(groups),
            "tier_counts_raw": dict(tier_counts),
            "tier_counts_grouped": dict(group_tier_counts),
            "instance_bbox_count": len(bbox_entries),
        },
        "groups": groups,
        "findings": findings,
    }

    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results written to {args.output}")

        # Tier distribution
        print("\n=== Tier distribution (groups) ===")
        for tier in ["high", "medium", "low", "info"]:
            n = group_tier_counts.get(tier, 0)
            if n:
                print(f"  {tier:>6}: {n} groups")

        # Top groups (skip info if any non-info exist, otherwise show info)
        non_info = [g for g in groups if g["severity_tier"] != "info"]
        show = non_info[:15] if non_info else groups[:15]
        title = "Top 15 actionable groups" if non_info else "Top 15 groups (all 'info' — no analog-name match yet)"
        print(f"\n=== {title} ===")
        for g in show:
            print(f"  [{g['severity_tier']:>6}] {g['check_layer']:>14}  x{g['count']:>3}  "
                  f"area={g['total_overlap_area']:>10.1f}  {g['owner_path']}")

        # Unique leaf cell distribution — useful for refining severity patterns
        leaf_counter = Counter()
        leaf_area = defaultdict(float)
        for g in groups:
            leaf = (g["owner_path"] or "").rsplit("/", 1)[-1] if g["owner_path"] else "(unresolved)"
            leaf_counter[leaf] += g["count"]
            leaf_area[leaf] += g["total_overlap_area"]
        print("\n=== Top 20 leaf cells by overlap count (use these to refine severity patterns) ===")
        for leaf, cnt in leaf_counter.most_common(20):
            print(f"  {cnt:>5} overlaps  area={leaf_area[leaf]:>10.1f}  {leaf}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
