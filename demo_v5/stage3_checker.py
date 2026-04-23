#!/usr/bin/env python3
"""
stage3_checker.py — stage 3 of the v5 pipeline.

Takes the 2C candidates from semantics.json and turns each (aggressor,
victim, target_cells, mechanism) triple into CONCRETE geometric hits
with world-space coordinates:

  for each candidate:
      a_polys = flatten top cell's aggressor-layer polygons to world coords
      v_polys = flatten top cell's victim-layer polygons to world coords
      clip both to each target-cell instance's world bbox
      STRtree intersect; for each overlap above min_area emit a hit

Output — issues.json:

  {
    "chip": { ... echo of extraction["chip"] ... },
    "source": {
      "extraction": "extraction.json",
      "semantics":  "semantics.json",
      "gds":        "chip.gds"
    },
    "min_overlap_area_um2": 0.5,
    "hits": [
      {
        "mechanism":         "rdl_over_bjt_pair",
        "severity":          "critical",
        "aggressor_layer":   "50/0",
        "victim_layer":      "82/5",
        "cell_path":         ["mpq8897r4_1", "BANDGAP", "BJT_pair"],
        "target_cell":       "BJT_pair",
        "overlap_bbox":      [x0, y0, x1, y1],
        "overlap_area_um2":  12.7,
        "reasoning":         "from 2C candidate #3"
      }
    ],
    "summary": {
      "n_candidates":        6,
      "n_hits":              23,
      "critical":            4,
      "high":                11,
      "medium":              8,
      "low":                 0,
      "by_mechanism":        { "rdl_over_bjt_pair": 4, ... }
    }
  }

Stage 3 is fully deterministic — no LLM. Given the same inputs it
produces the same output.

Usage:
    python demo_v5/stage3_checker.py demo_v5/extraction.json
    python demo_v5/stage3_checker.py demo_v5/extraction.json \
        --semantics demo_v5/semantics.json \
        --output demo_v5/issues.json \
        --gds /path/to/chip.gds
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import gdstk
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.strtree import STRtree
from shapely.validation import make_valid

# Reuse stage-1 transform helpers so stage 1 and stage 3 agree on world coords
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor import (  # noqa: E402
    IDENTITY, ref_transform, compose, transform_bbox, walk_instance_bboxes,
)


MIN_OVERLAP_AREA_UM2 = 0.5  # same floor as demo_complex/layer_config.json


# ── Layer id parsing ────────────────────────────────────────────────────

def parse_ldt(layer_id):
    """'50/0' → (50, 0). Returns None on malformed input."""
    if not layer_id or "/" not in str(layer_id):
        return None
    try:
        l, d = layer_id.split("/", 1)
        return (int(l), int(d))
    except (ValueError, AttributeError):
        return None


# ── World-space polygon extraction ──────────────────────────────────────

def extract_world_polygons(top_cell, ldt):
    """
    Flatten every polygon on (layer, datatype) under top_cell into world
    coordinates. gdstk's get_polygons() already composes transforms for us.
    Returns a list of ShapelyPolygons.
    """
    out = []
    for gp in top_cell.get_polygons(layer=ldt[0], datatype=ldt[1]):
        pts = gp.points if isinstance(gp, gdstk.Polygon) else gp
        if len(pts) < 3:
            continue
        sp = ShapelyPolygon(pts)
        if not sp.is_valid:
            sp = make_valid(sp)
            if sp.is_empty or sp.geom_type not in ("Polygon", "MultiPolygon"):
                continue
        if sp.is_empty or sp.area == 0:
            continue
        if sp.geom_type == "MultiPolygon":
            out.extend(g for g in sp.geoms if not g.is_empty and g.area > 0)
        else:
            out.append(sp)
    return out


# ── Target-cell bbox resolution ─────────────────────────────────────────

def target_bboxes(bbox_entries, target_cells):
    """
    Return [(shapely_box, path_tuple, target_cell_name)] for every instance
    of any target cell in the hierarchy. If target_cells is falsy/empty,
    returns [] and the caller should fall back to chip-wide check.
    """
    if not target_cells:
        return []
    target_set = set(target_cells)
    out = []
    for sh, path, _depth in bbox_entries:
        # The cell-of-interest is the last element in the path
        if path[-1] in target_set:
            out.append((sh, path, path[-1]))
    return out


# ── Overlap computation per candidate ───────────────────────────────────

def check_candidate(candidate, top_cell, bbox_entries, chip_bbox,
                    min_area=MIN_OVERLAP_AREA_UM2,
                    poly_cache=None):
    """
    Given one 2C candidate, return a list of concrete hits.
    """
    a_ldt = parse_ldt(candidate.get("aggressor_layer"))
    v_ldt = parse_ldt(candidate.get("victim_layer"))
    if a_ldt is None or v_ldt is None:
        return [{"skipped": True, "reason": "malformed layer id",
                 "candidate": candidate}]

    # Cache per-layer world polygons across candidates so we flatten each
    # layer at most once even if several candidates share an aggressor.
    if poly_cache is None:
        poly_cache = {}
    if a_ldt not in poly_cache:
        poly_cache[a_ldt] = extract_world_polygons(top_cell, a_ldt)
    if v_ldt not in poly_cache:
        poly_cache[v_ldt] = extract_world_polygons(top_cell, v_ldt)
    a_polys = poly_cache[a_ldt]
    v_polys = poly_cache[v_ldt]

    if not a_polys or not v_polys:
        return [{"skipped": True,
                 "reason": f"no polygons on {candidate.get('aggressor_layer')} "
                           f"or {candidate.get('victim_layer')}",
                 "candidate": candidate}]

    # Resolve target regions. If none matched, fall back to the chip bbox so
    # the check still runs — but flag it so the copilot can point out that
    # the target_cells list didn't match any instantiated cell.
    tgts = target_bboxes(bbox_entries, candidate.get("target_cells") or [])
    fallback_used = False
    if not tgts:
        tgts = [(chip_bbox, ("chip",), "chip")]
        fallback_used = True

    # Build STRtree of victim polygons once for this candidate
    v_tree = STRtree(v_polys)

    hits = []
    seen_overlaps = set()  # de-dup by (cell_path, rounded-overlap-bbox)

    for tgt_bb, tgt_path, tgt_name in tgts:
        tx0, ty0, tx1, ty1 = tgt_bb.bounds
        tgt_shape = tgt_bb

        # Narrow aggressors to ones that touch this target bbox
        a_in_tgt = [p for p in a_polys if p.intersects(tgt_shape)]
        if not a_in_tgt:
            continue

        for a in a_in_tgt:
            # Clip aggressor to the target region first so overlap area is
            # measured only inside the cell instance we care about.
            a_clip = a.intersection(tgt_shape)
            if a_clip.is_empty or a_clip.area == 0:
                continue

            for vi in v_tree.query(a_clip):
                v = v_polys[int(vi)]
                if not a_clip.intersects(v):
                    continue
                try:
                    ovlp = a_clip.intersection(v)
                except Exception:
                    continue
                if ovlp.is_empty or ovlp.area < min_area:
                    continue

                # Split MultiPolygon into separate hits so bboxes are tight
                parts = [ovlp] if ovlp.geom_type == "Polygon" else list(
                    getattr(ovlp, "geoms", [ovlp]))
                for part in parts:
                    if part.is_empty or part.area < min_area:
                        continue
                    if part.geom_type not in ("Polygon",):
                        continue
                    ox0, oy0, ox1, oy1 = part.bounds
                    key = (tgt_path,
                           round(ox0, 2), round(oy0, 2),
                           round(ox1, 2), round(oy1, 2))
                    if key in seen_overlaps:
                        continue
                    seen_overlaps.add(key)
                    hits.append({
                        "mechanism": candidate.get("mechanism", "unknown"),
                        "severity": candidate.get("severity", "unknown"),
                        "aggressor_layer": candidate["aggressor_layer"],
                        "victim_layer": candidate["victim_layer"],
                        "cell_path": list(tgt_path),
                        "target_cell": tgt_name,
                        "overlap_bbox": [float(ox0), float(oy0),
                                         float(ox1), float(oy1)],
                        "overlap_area_um2": float(part.area),
                        "reasoning": candidate.get("reasoning", ""),
                        "fallback_chipwide": fallback_used,
                    })

    return hits


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("extraction_json")
    ap.add_argument("--semantics", default=None,
                    help="Default: semantics.json next to extraction.json")
    ap.add_argument("--output", default=None,
                    help="Default: issues.json next to extraction.json")
    ap.add_argument("--gds", default=None,
                    help="Override GDS path (otherwise read from extraction.json)")
    ap.add_argument("--min-area", type=float, default=MIN_OVERLAP_AREA_UM2,
                    help=f"Minimum overlap area in um^2. Default {MIN_OVERLAP_AREA_UM2}.")
    args = ap.parse_args()

    t_start = time.time()

    # ── load extraction + semantics ─────────────────────────────────────
    if not os.path.exists(args.extraction_json):
        sys.exit(f"Not found: {args.extraction_json}")
    extraction = json.load(open(args.extraction_json))

    sem_path = args.semantics or os.path.join(
        os.path.dirname(os.path.abspath(args.extraction_json)), "semantics.json")
    if not os.path.exists(sem_path):
        sys.exit(f"Not found: {sem_path}. Run stage 2 first.")
    semantics = json.load(open(sem_path))

    candidates_section = semantics.get("2c_candidates") or {}
    if "error" in candidates_section:
        sys.exit(f"Stage 2C failed — cannot run stage 3. "
                 f"Error: {candidates_section.get('error')}")
    candidates = candidates_section.get("candidates", [])
    if not candidates:
        print("No 2C candidates — writing empty issues.json.", file=sys.stderr)

    # ── locate the GDS file ─────────────────────────────────────────────
    gds_path = args.gds or extraction.get("gds_file_path") or extraction.get("gds_file")
    if not gds_path or not os.path.exists(gds_path):
        sys.exit(f"GDS file not found: {gds_path}. Pass --gds to override.")

    print(f"Loading {gds_path} ...")
    t0 = time.time()
    lib = gdstk.read_gds(gds_path)
    print(f"  {len(lib.cells)} cells in {time.time()-t0:.1f}s")

    top_name = extraction["chip"]["top_cell"]
    top_cell = next((c for c in lib.cells if c.name == top_name), None)
    if top_cell is None:
        sys.exit(f"Top cell '{top_name}' not found in GDS.")

    dbb = extraction["chip"]["die_bbox"]
    chip_bbox = shapely_box(dbb[0], dbb[1], dbb[2], dbb[3])

    # ── walk hierarchy for target-cell instance bboxes ──────────────────
    print("Walking hierarchy ...")
    t0 = time.time()
    bbox_entries = walk_instance_bboxes(top_cell)
    print(f"  {len(bbox_entries)} instance bboxes in {time.time()-t0:.1f}s")

    # ── run each candidate ──────────────────────────────────────────────
    print(f"\nRunning {len(candidates)} candidate checks "
          f"(min overlap {args.min_area} um^2)")
    poly_cache = {}
    all_hits = []
    skipped = []
    for i, cand in enumerate(candidates, 1):
        mech = cand.get("mechanism", "?")
        sev = cand.get("severity", "?")
        print(f"  [{i}/{len(candidates)}] {mech} ({sev}) — "
              f"{cand.get('aggressor_layer')} → {cand.get('victim_layer')}")
        t0 = time.time()
        results = check_candidate(cand, top_cell, bbox_entries, chip_bbox,
                                  min_area=args.min_area, poly_cache=poly_cache)
        hits = [r for r in results if not r.get("skipped")]
        sk = [r for r in results if r.get("skipped")]
        all_hits.extend(hits)
        skipped.extend(sk)
        print(f"       {len(hits)} hits in {time.time()-t0:.1f}s"
              + (f"  (skipped: {sk[0]['reason']})" if sk else ""))

    # ── summary ─────────────────────────────────────────────────────────
    by_mech = defaultdict(int)
    by_sev = defaultdict(int)
    for h in all_hits:
        by_mech[h["mechanism"]] += 1
        by_sev[h["severity"]] += 1

    out = {
        "schema_version": 1,
        "elapsed_s": round(time.time() - t_start, 2),
        "chip": extraction["chip"],
        "source": {
            "extraction": os.path.abspath(args.extraction_json),
            "semantics": os.path.abspath(sem_path),
            "gds": os.path.abspath(gds_path),
        },
        "min_overlap_area_um2": args.min_area,
        "hits": all_hits,
        "skipped_candidates": skipped,
        "summary": {
            "n_candidates": len(candidates),
            "n_hits": len(all_hits),
            "n_skipped_candidates": len(skipped),
            "critical": by_sev.get("critical", 0),
            "high": by_sev.get("high", 0),
            "medium": by_sev.get("medium", 0),
            "low": by_sev.get("low", 0),
            "by_mechanism": dict(by_mech),
        },
    }

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.extraction_json)), "issues.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    size_kb = os.path.getsize(out_path) / 1024

    print(f"\nDone. {len(all_hits)} hits across {len(candidates)} candidates "
          f"in {out['elapsed_s']}s total.")
    print(f"  severity:   critical={by_sev.get('critical', 0)}  "
          f"high={by_sev.get('high', 0)}  medium={by_sev.get('medium', 0)}  "
          f"low={by_sev.get('low', 0)}")
    print(f"  by mechanism: {dict(by_mech)}")
    print(f"  wrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
