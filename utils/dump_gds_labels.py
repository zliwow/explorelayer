#!/usr/bin/env python3
"""
Inspect GDS text labels — which layers they live on, what the text looks like,
and whether they match chip-top pin names.

Usage:
    python dump_gds_labels.py examples/mpq8897r4_1.gds
    python dump_gds_labels.py examples/mpq8897r4_1.gds --cell mpq8897r4_1
    python dump_gds_labels.py examples/mpq8897r4_1.gds --pins top_body.txt --output labels_report.txt

If --pins is given (path to a subckt dump with pin names), the script
reports how many labels literally match a chip-top pin name.
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import gdstk


def load_pin_names(pins_file):
    """Parse pin names out of a dump_subckt.py report (lines between '=== Pins ===' and next '===')."""
    if not pins_file or not os.path.exists(pins_file):
        return set()
    pins = set()
    in_pins = False
    with open(pins_file, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("=== Pins ==="):
                in_pins = True
                continue
            if in_pins and s.startswith("===") and not s.startswith("=== Pins"):
                break
            if in_pins and s:
                # Pin rows are whitespace-separated names; drop empties
                for tok in s.split():
                    # strip any stray indicators
                    if tok and tok != "===":
                        pins.add(tok)
    return pins


def collect_labels(lib, cell_name=None):
    """Walk every cell (so labels inside children are counted) and return list of (layer, dt, text, x, y, cell)."""
    results = []
    target_cells = [c for c in lib.cells if not cell_name or c.name == cell_name]
    # If a top cell was specified, traverse the whole hierarchy beneath it too.
    if cell_name:
        top = next((c for c in lib.cells if c.name == cell_name), None)
        if top is None:
            print(f"Error: cell '{cell_name}' not found.", file=sys.stderr)
            sys.exit(1)
        seen = set()
        stack = [top]
        target_cells = []
        while stack:
            c = stack.pop()
            if c.name in seen:
                continue
            seen.add(c.name)
            target_cells.append(c)
            for ref in c.references:
                ref_cell = ref.cell if isinstance(ref.cell, gdstk.Cell) else None
                if ref_cell is not None:
                    stack.append(ref_cell)

    for c in target_cells:
        for lbl in c.labels:
            try:
                x, y = float(lbl.origin[0]), float(lbl.origin[1])
            except Exception:
                x, y = 0.0, 0.0
            results.append((int(lbl.layer), int(lbl.texttype), lbl.text, x, y, c.name))
    return results


def pattern_signature(text):
    """
    Reduce a label string to a coarse pattern so we can cluster similar names.
      'LabelToLayout_1234' -> 'LabelToLayout_#'
      'VBIAS_pad'          -> 'VBIAS_pad'
      'mtp_addr<6>'        -> 'mtp_addr<#>'
    """
    t = re.sub(r"\d+", "#", text)
    return t


def report(labels, pin_names):
    out = []
    out.append(f"=== GDS label inspection ===")
    out.append(f"Total labels: {len(labels):,}")
    out.append("")

    # Per-layer breakdown
    per_layer = Counter((l, d) for l, d, *_ in labels)
    out.append("=== Labels per (layer, datatype) ===")
    for (layer, dt), count in sorted(per_layer.items(), key=lambda x: -x[1]):
        out.append(f"  layer {layer}/{dt}: {count:,}")
    out.append("")

    # Pattern clustering
    patterns = Counter(pattern_signature(t) for _, _, t, *_ in labels)
    out.append("=== Top 20 label text patterns (digits replaced with #) ===")
    for pat, c in patterns.most_common(20):
        out.append(f"  {c:6,}  {pat!r}")
    out.append("")

    # Sample real labels per layer
    by_layer_samples = defaultdict(list)
    for layer, dt, text, x, y, cell in labels:
        if len(by_layer_samples[(layer, dt)]) < 5:
            by_layer_samples[(layer, dt)].append((text, x, y, cell))
    out.append("=== 5 sample labels per layer ===")
    for (layer, dt) in sorted(by_layer_samples):
        out.append(f"-- layer {layer}/{dt} --")
        for text, x, y, cell in by_layer_samples[(layer, dt)]:
            out.append(f"  '{text}'   at ({x:.2f}, {y:.2f}) in cell {cell}")
    out.append("")

    # Overlap with pin names
    if pin_names:
        label_texts = {t for _, _, t, *_ in labels}
        exact_matches = label_texts & pin_names
        # Also try a looser match: strip common suffixes like _pad or <idx>
        def norm(s):
            s = s.strip().lower()
            s = re.sub(r"_pad$", "", s)
            s = re.sub(r"<\d+>", "", s)
            return s
        pin_norm = {norm(p): p for p in pin_names}
        lbl_norm = {norm(t): t for t in label_texts}
        fuzzy_matches = set(pin_norm) & set(lbl_norm)

        out.append("=== Match against chip-top pin names ===")
        out.append(f"  Unique label texts:            {len(label_texts):,}")
        out.append(f"  Unique chip-top pins:          {len(pin_names):,}")
        out.append(f"  Exact label-text == pin-name:  {len(exact_matches):,}")
        out.append(f"  Fuzzy (strip _pad, <idx>):     {len(fuzzy_matches):,}")
        if exact_matches:
            sample = sorted(exact_matches)[:20]
            out.append(f"  Sample exact matches: {', '.join(sample)}")
        else:
            out.append("  NO exact matches — labels probably aren't real net names.")
        out.append("")

    out.append("=== Verdict ===")
    # Heuristic: if the single most common pattern covers >70%, labels are likely auto-IDs
    if labels:
        top_pat, top_count = patterns.most_common(1)[0]
        frac = top_count / len(labels)
        if frac > 0.7:
            out.append(f"  {frac:.0%} of labels match pattern {top_pat!r} — likely auto-generated")
            out.append(f"  probe IDs (Calibre/Argus LabelToLayout). Not usable as net names.")
        elif pin_names and len((label_texts := {t for _, _, t, *_ in labels}) & pin_names) > 20:
            out.append(f"  {len(label_texts & pin_names)} labels match chip-top pin names —")
            out.append(f"  usable as pad->net mapping. Proceed with label-based overlap tagging.")
        else:
            out.append(f"  Mixed / unclear. Look at the sample labels above to decide.")
    out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Inspect GDS text labels.")
    p.add_argument("gds")
    p.add_argument("--cell", default=None, help="Target top cell (default: all cells)")
    p.add_argument("--pins", default=None, help="Path to a dump_subckt.py report — label texts will be compared to its pin list")
    p.add_argument("--output", default=None, help="Write report to file instead of stdout")
    args = p.parse_args()

    if not os.path.exists(args.gds):
        print(f"Error: file not found: {args.gds}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.gds} ...", file=sys.stderr)
    lib = gdstk.read_gds(args.gds)
    print(f"  {len(lib.cells)} cells", file=sys.stderr)

    print("Collecting labels ...", file=sys.stderr)
    labels = collect_labels(lib, args.cell)
    print(f"  {len(labels):,} labels", file=sys.stderr)

    pin_names = load_pin_names(args.pins) if args.pins else set()
    if args.pins:
        print(f"  {len(pin_names):,} pin names loaded from {args.pins}", file=sys.stderr)

    text = report(labels, pin_names)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
