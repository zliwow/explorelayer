#!/usr/bin/env python3
"""
inspect_extraction.py — print a human-readable summary of extraction.json.

Lets you eyeball the signals (without opening a 400 KB JSON) before
handing it to the stage-2 LLM reasoner. Also prints section byte-sizes
so you can see what's bloating the file.

Usage:
    python demo_v5/inspect_extraction.py demo_v5/extraction.json
    python demo_v5/inspect_extraction.py demo_v5/extraction.json --top 20
"""

import argparse
import json
import os
import sys


def section_bytes(obj):
    return len(json.dumps(obj))


def fmt_kb(n):
    return f"{n/1024:.1f} KB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("json_file")
    ap.add_argument("--top", type=int, default=15,
                    help="Rows to show in each ranked table. Default 15.")
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        sys.exit(f"Not found: {args.json_file}")

    d = json.load(open(args.json_file))
    total_kb = os.path.getsize(args.json_file) / 1024

    print(f"\n========== {args.json_file}  ({total_kb:.1f} KB) ==========\n")

    # ── size breakdown ──────────────────────────────────────────────────
    print("-- section sizes --")
    for k in d:
        size = section_bytes(d[k])
        pct = size / (total_kb * 1024) * 100
        print(f"  {k:26s}  {fmt_kb(size):>10s}   {pct:5.1f}%")
    print()

    # ── chip basics ─────────────────────────────────────────────────────
    c = d.get("chip", {})
    print("-- chip --")
    print(f"  top cell          {c.get('top_cell')}")
    print(f"  cells in library  {c.get('n_cells_in_library')}")
    print(f"  die               {c.get('die_w'):.1f} x {c.get('die_h'):.1f} um "
          f"(area {c.get('die_area'):.0f} um^2)")
    print(f"  hierarchy depth   {c.get('hierarchy_max_depth')}")
    print(f"  instance bboxes   {c.get('n_instance_bboxes')}")
    print(f"  total polygons    {c.get('total_polygons'):,}")
    print()

    # ── layer inventory ────────────────────────────────────────────────
    layers = d.get("layers", [])
    print(f"-- layers ({len(layers)} total) --")
    print(f"  {'id':>8s}  {'weak_cat':22s}  {'polys':>10s}  "
          f"{'mean_area':>10s}  {'die%':>6s}  {'rect%':>6s}  {'edge%':>6s}  "
          f"reasoning")
    cats_seen = {}
    for li in layers:
        s = li.get("stats") or {}
        cat = li.get("weak_category", "?")
        cats_seen[cat] = cats_seen.get(cat, 0) + 1
        print(f"  {li['id']:>8s}  {cat:22s}  "
              f"{s.get('total_count', 0):>10,}  "
              f"{s.get('mean_area', 0):>10.0f}  "
              f"{s.get('area_pct_of_die', 0):>6.1f}  "
              f"{s.get('pct_rect', 0):>6.0f}  "
              f"{s.get('pct_near_edge', 0):>6.0f}  "
              f"{li.get('weak_reasoning', '')}")
    print()
    print("  category counts:", dict(sorted(cats_seen.items(), key=lambda kv: -kv[1])))
    print()

    # ── top layer-pair overlaps ────────────────────────────────────────
    pairs = d.get("layer_pair_overlaps", [])
    print(f"-- layer-pair overlaps (top {args.top} of {len(pairs)}) --")
    print(f"  {'a':>8s}  {'b':>8s}  {'overlap um^2':>14s}  "
          f"{'a%':>7s}  {'b%':>7s}  note")
    for p in pairs[:args.top]:
        print(f"  {p['a']:>8s}  {p['b']:>8s}  "
              f"{p['overlap_area']:>14,.0f}  "
              f"{p['a_pct']:>6.1f}%  {p['b_pct']:>6.1f}%  {p.get('note','')}")
    print()

    # ── cells with semantic hints ──────────────────────────────────────
    cells = d.get("cells", [])
    hinted = [c for c in cells if c.get("name_hint")]
    print(f"-- cells with name hint ({len(hinted)} of {len(cells)} dumped) --")
    if hinted:
        print(f"  {'name':40s}  {'hint':28s}  {'own':>6s}  {'inst':>5s}  children")
        for c in hinted[:args.top]:
            kids = ",".join(c.get("children", [])[:4])
            if len(c.get("children", [])) > 4:
                kids += f",… (+{len(c['children']) - 4})"
            print(f"  {c['name'][:40]:40s}  {c['name_hint']:28s}  "
                  f"{c['own_polys']:>6,}  {c['n_instances']:>5d}  {kids}")
    else:
        print("  (none — real chip may need tuning of name_hint patterns)")
    print()

    # ── top cells by own polygons ──────────────────────────────────────
    print(f"-- top {args.top} cells by own polygons --")
    print(f"  {'name':40s}  {'own':>8s}  {'inst':>5s}  {'kids':>5s}  top hint")
    for c in cells[:args.top]:
        print(f"  {c['name'][:40]:40s}  {c['own_polys']:>8,}  "
              f"{c['n_instances']:>5d}  {c['n_children']:>5d}  "
              f"{c.get('name_hint') or ''}")
    print()

    # ── labels ──────────────────────────────────────────────────────────
    labels = d.get("labels_by_layer", {})
    n_labels = sum(len(v) for v in labels.values())
    print(f"-- labels ({n_labels} total across {len(labels)} layer/texttype pairs) --")
    for layer_id, labs in labels.items():
        print(f"  layer {layer_id}:  {len(labs)} labels")
        samples = [l["text"] for l in labs[:20]]
        print(f"    sample: {samples}")
    print()

    # ── pad-like shape counts per layer ────────────────────────────────
    pads = d.get("pad_shapes_by_layer", {})
    print(f"-- pad-like shape counts --")
    for k, v in pads.items():
        print(f"  {k}:  {len(v)} merged shapes")
    print()


if __name__ == "__main__":
    main()
