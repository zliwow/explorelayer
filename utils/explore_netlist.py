#!/usr/bin/env python3
"""
Netlist Explorer — scan a CDL/SPICE/Spectre netlist and print a text report
you can paste into chat for parser design.

Usage:
    python explore_netlist.py mpq8897_esd_r4.cdl
    python explore_netlist.py mpq8897_esd_r4.cdl --output netlist_summary.txt
    python explore_netlist.py some_block.netlist --sample-devices 20
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict


# ── Dialect detection ────────────────────────────────────────────────────────

def detect_dialect(first_lines):
    """Guess CDL / HSPICE / Spectre from the first few hundred lines."""
    blob = "\n".join(first_lines).lower()
    if "simulator lang=spectre" in blob or "//" in blob[:2000]:
        return "Spectre"
    if ".subckt" in blob and "*.cdl" in blob:
        return "CDL (Cadence, explicit header)"
    if ".subckt" in blob:
        # Could be either CDL or HSPICE. CDL is usually plainer.
        if re.search(r"\.(tran|ac|dc|op|meas|probe)", blob):
            return "HSPICE (simulation deck)"
        return "CDL / HSPICE (SPICE-family)"
    return "Unknown (no .SUBCKT found in header)"


# ── Main scan ────────────────────────────────────────────────────────────────

DEVICE_PREFIXES = {
    "M": "MOSFETs",
    "R": "Resistors",
    "C": "Capacitors",
    "L": "Inductors",
    "D": "Diodes",
    "Q": "BJTs",
    "J": "JFETs",
    "K": "Coupled inductors",
    "V": "Voltage sources",
    "I": "Current sources",
    "X": "Subckt instances",
}

# Match device model name heuristically: last non-param token before params
PARAM_RE = re.compile(r"^\w+\s*=")


def scan(path, sample_n=5):
    stats = {
        "file": path,
        "size_mb": os.path.getsize(path) / (1024 * 1024),
        "total_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "continuation_lines": 0,
        "max_line_len": 0,
        "subckt_defs": [],          # list of (name, pins, line_no)
        "subckt_instances": Counter(),  # which subckts are X-referenced
        "device_counts": Counter(),
        "model_counts": Counter(),
        "samples": defaultdict(list),
        "warnings": [],
        "first_lines": [],
    }

    current_subckt = None
    in_subckt = False

    with open(path, "r", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            stats["total_lines"] = lineno
            line = raw.rstrip("\n")
            stripped = line.strip()

            # Grab first 200 lines for dialect detection and header view
            if lineno <= 200:
                stats["first_lines"].append(line)

            if len(line) > stats["max_line_len"]:
                stats["max_line_len"] = len(line)

            if not stripped:
                stats["blank_lines"] += 1
                continue

            first = stripped[0]

            # Comments
            if first == "*" or stripped.startswith("//") or first == "$":
                stats["comment_lines"] += 1
                continue

            # Continuation line
            if first == "+":
                stats["continuation_lines"] += 1
                continue

            # Case-insensitive directives
            low = stripped.lower()

            # Subckt definition
            if low.startswith(".subckt") or low.startswith("subckt "):
                parts = stripped.split()
                if len(parts) >= 2:
                    name = parts[1]
                    pins = [p for p in parts[2:] if "=" not in p]
                    stats["subckt_defs"].append((name, pins, lineno))
                    current_subckt = name
                    in_subckt = True
                continue

            if low.startswith(".ends") or low == "ends":
                in_subckt = False
                current_subckt = None
                continue

            # Other SPICE directives — skip
            if first == ".":
                continue

            # Device line: first char is device-type letter
            dev_type = first.upper()
            if dev_type in DEVICE_PREFIXES:
                stats["device_counts"][dev_type] += 1

                # Keep a few samples per type
                if len(stats["samples"][dev_type]) < sample_n:
                    stats["samples"][dev_type].append((lineno, stripped[:200]))

                # Extract model name (for M/R/C/D/Q/J/X)
                tokens = stripped.split()
                if dev_type == "X":
                    # X<name> <pin1> ... <pinN> <subckt_name>
                    # Subckt name is the last non-param token
                    non_param = [t for t in tokens[1:] if not PARAM_RE.match(t)]
                    if non_param:
                        sub = non_param[-1]
                        stats["subckt_instances"][sub] += 1
                        stats["model_counts"][f"X:{sub}"] += 1
                elif dev_type in ("M", "R", "C", "D", "Q", "J"):
                    # Model name is last non-param token
                    non_param = [t for t in tokens[1:] if not PARAM_RE.match(t)]
                    if non_param:
                        model = non_param[-1]
                        stats["model_counts"][f"{dev_type}:{model}"] += 1

    return stats


# ── Report ───────────────────────────────────────────────────────────────────

def fmt_int(n):
    return f"{n:,}"


def pick_top_level(subckt_defs, subckt_instances):
    """
    Top-level subckt = defined but never instantiated (nothing X-references it).
    If multiple, pick the one with the most pins (chip-top usually has many).
    """
    defined = {name for name, _, _ in subckt_defs}
    used = set(subckt_instances.keys())
    roots = defined - used
    if not roots:
        return None
    # Pick by pin count
    pin_counts = {name: len(pins) for name, pins, _ in subckt_defs if name in roots}
    return max(pin_counts, key=pin_counts.get)


def report(stats, sample_n=5):
    out = []
    f = stats["file"]

    out.append(f"=== Netlist Explorer: {os.path.basename(f)} ===")
    out.append(f"Path:              {f}")
    out.append(f"Size:              {stats['size_mb']:.1f} MB")
    out.append(f"Total lines:       {fmt_int(stats['total_lines'])}")
    out.append(f"  comments:        {fmt_int(stats['comment_lines'])}")
    out.append(f"  blank:           {fmt_int(stats['blank_lines'])}")
    out.append(f"  continuations:   {fmt_int(stats['continuation_lines'])}")
    out.append(f"Max line length:   {stats['max_line_len']}")

    dialect = detect_dialect(stats["first_lines"])
    out.append(f"Dialect guess:     {dialect}")
    out.append("")

    # Structure
    out.append("=== Structure ===")
    out.append(f"Subckts defined:   {fmt_int(len(stats['subckt_defs']))}")
    out.append(f"Unique subckts X-referenced: {fmt_int(len(stats['subckt_instances']))}")

    top = pick_top_level(stats["subckt_defs"], stats["subckt_instances"])
    if top:
        top_def = next(d for d in stats["subckt_defs"] if d[0] == top)
        _, pins, lineno = top_def
        out.append(f"Top-level subckt:  {top} ({len(pins)} pins, defined at line {lineno})")
        out.append(f"  first 20 pins:   {' '.join(pins[:20])}")
        if len(pins) > 20:
            out.append(f"  ... and {len(pins)-20} more")
    else:
        out.append("Top-level subckt:  (could not identify — all subckts are instantiated)")
    out.append("")

    # Device census
    out.append("=== Device census ===")
    total_devices = sum(stats["device_counts"].values())
    out.append(f"Total device lines: {fmt_int(total_devices)}")
    for prefix, label in DEVICE_PREFIXES.items():
        count = stats["device_counts"].get(prefix, 0)
        if count > 0:
            out.append(f"  {label:22s} ({prefix}): {fmt_int(count)}")
    out.append("")

    # Top models
    out.append("=== Top device models (top 20 by count) ===")
    for model, count in stats["model_counts"].most_common(20):
        out.append(f"  {model:40s} {fmt_int(count)}")
    out.append("")

    # Top X-instantiated subckts
    out.append("=== Most-instantiated subckts (top 15) ===")
    for sub, count in stats["subckt_instances"].most_common(15):
        out.append(f"  {sub:40s} {fmt_int(count)}")
    out.append("")

    # Sample device lines
    out.append(f"=== Sample device lines ({sample_n} per type) ===")
    for prefix in ["M", "R", "C", "D", "Q", "X"]:
        samples = stats["samples"].get(prefix, [])
        if not samples:
            continue
        out.append(f"-- {DEVICE_PREFIXES[prefix]} ({prefix}) --")
        for lineno, text in samples:
            out.append(f"  L{lineno}: {text}")
    out.append("")

    # First real lines (skipping comments) for dialect sanity
    out.append("=== First 20 non-comment, non-blank lines ===")
    shown = 0
    for line in stats["first_lines"]:
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("//"):
            continue
        out.append(f"  {line[:180]}")
        shown += 1
        if shown >= 20:
            break
    out.append("")

    # Warnings
    if stats["warnings"]:
        out.append("=== Warnings ===")
        for w in stats["warnings"]:
            out.append(f"  - {w}")
        out.append("")

    # Ballpark interpretation
    out.append("=== Interpretation hints ===")
    mos = stats["device_counts"].get("M", 0)
    if mos > 100_000:
        out.append("  - Full chip netlist (MOSFET count > 100k)")
    elif mos > 10_000:
        out.append("  - Large block or sub-system")
    elif mos > 100:
        out.append("  - Small block or sub-module")
    else:
        out.append("  - Very small — testbench or unit cell")

    if top and "mpq" in top.lower():
        out.append(f"  - Top cell '{top}' matches chip name — likely the one we want")
    out.append("")

    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore a CDL/SPICE netlist.")
    parser.add_argument("netlist_file", help="Path to the .cdl / .sp / .netlist file")
    parser.add_argument("--output", default=None, help="Write report to file instead of stdout")
    parser.add_argument("--sample-devices", type=int, default=5,
                        help="Number of sample device lines per type to include (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.netlist_file):
        print(f"Error: file not found: {args.netlist_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.netlist_file} ...", file=sys.stderr)
    stats = scan(args.netlist_file, sample_n=args.sample_devices)
    text = report(stats, sample_n=args.sample_devices)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
