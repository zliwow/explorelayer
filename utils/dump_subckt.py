#!/usr/bin/env python3
"""
Dump one subckt's pins, internal nets, and instance census from a CDL/SPICE netlist.

Used after find_top_subckt.py has identified the chip-top subckt — this script
extracts that subckt's body so we know its pin list and what sub-modules sit
directly under it. Handles SPICE '+' continuation lines.

Usage:
    python dump_subckt.py examples/MPQ8897_ESD_R4.cdl --name MPQ8897_TOP_R4
    python dump_subckt.py examples/MPQ8897_ESD_R4.cdl --name MPQ8897_TOP_R4 --output mpq8897_top_r4_body.txt
    python dump_subckt.py examples/MPQ8897_ESD_R4.cdl --name MPQ8897_TOP_R4 --raw raw_body.sp
"""

import argparse
import os
import re
import sys
from collections import Counter

PARAM_RE = re.compile(r"^\w+\s*=")
DEVICE_PREFIXES = {
    "M": "MOSFETs", "R": "Resistors", "C": "Capacitors",
    "L": "Inductors", "D": "Diodes", "Q": "BJTs", "J": "JFETs",
    "K": "Coupled inductors", "V": "V-sources", "I": "I-sources",
    "X": "Subckt instances",
}


def find_and_extract(path, target_name):
    """
    Return (start_line, end_line, header_tokens, body_lines) for the subckt.
    Body lines are joined with continuations ('+') already folded in.
    """
    target_lc = target_name.lower()
    start = None
    end = None
    header_raw = []
    body_raw = []  # (lineno, joined_text)

    current_line = None
    current_lineno = None

    def flush():
        nonlocal current_line, current_lineno
        if current_line is not None:
            body_raw.append((current_lineno, current_line))
            current_line = None
            current_lineno = None

    with open(path, "r", errors="replace") as f:
        in_target = False
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            low = stripped.lower()

            if not in_target:
                if low.startswith(".subckt") or low.startswith("subckt "):
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[1].lower() == target_lc:
                        in_target = True
                        start = lineno
                        # Collect header (including continuation lines)
                        header_raw.append(line)
                continue

            # In target — check for continuation of header
            if header_raw and stripped.startswith("+"):
                header_raw.append(line)
                continue

            # Header is done once we see a non-'+' line
            # Now accumulate body lines with continuation folding
            if low.startswith(".ends") or low == "ends":
                flush()
                end = lineno
                break

            if stripped.startswith("+"):
                if current_line is None:
                    # orphan continuation, skip
                    continue
                current_line = current_line + " " + stripped[1:].strip()
                continue

            # New statement
            flush()
            if stripped:
                current_line = stripped
                current_lineno = lineno

        flush()

    if start is None:
        return None

    # Parse header tokens (everything after .subckt <name>)
    header_joined = " ".join(
        h.strip().lstrip("+").strip() for h in header_raw
    )
    # Strip leading .subckt and the name
    m = re.match(r"\.?subckt\s+\S+\s*(.*)", header_joined, flags=re.IGNORECASE)
    header_rest = m.group(1) if m else ""
    header_tokens = header_rest.split()
    pins = [t for t in header_tokens if "=" not in t]
    params = [t for t in header_tokens if "=" in t]

    return {
        "start": start,
        "end": end,
        "pins": pins,
        "params": params,
        "body": body_raw,
    }


def analyze_body(body_raw):
    device_counts = Counter()
    model_counts = Counter()
    subckt_instances = Counter()
    nets = set()
    samples = {k: [] for k in DEVICE_PREFIXES}

    for lineno, text in body_raw:
        if not text or text[0] == "*" or text.startswith("//"):
            continue
        if text[0] == ".":
            continue  # nested directives — ignore

        first = text[0].upper()
        if first not in DEVICE_PREFIXES:
            continue

        device_counts[first] += 1
        tokens = text.split()

        if len(samples[first]) < 3:
            samples[first].append((lineno, text[:180]))

        # Collect nets: skip param tokens, skip instance name (tokens[0]),
        # skip model/subckt name (last non-param token)
        non_param = [t for t in tokens[1:] if not PARAM_RE.match(t)]
        if len(non_param) < 2:
            continue
        model = non_param[-1]
        pin_nets = non_param[:-1]

        nets.update(pin_nets)

        if first == "X":
            subckt_instances[model] += 1
            model_counts[f"X:{model}"] += 1
        elif first in ("M", "R", "C", "D", "Q", "J"):
            model_counts[f"{first}:{model}"] += 1

    return {
        "device_counts": device_counts,
        "model_counts": model_counts,
        "subckt_instances": subckt_instances,
        "nets": nets,
        "samples": samples,
    }


def report(path, name, info, stats):
    out = []
    out.append(f"=== Subckt dump: {name} ===")
    out.append(f"File:         {os.path.basename(path)}")
    out.append(f"Defined at:   line {info['start']}")
    out.append(f"Ends at:      line {info['end']}")
    out.append(f"Pin count:    {len(info['pins'])}")
    out.append(f"Param count:  {len(info['params'])}")
    out.append("")

    out.append("=== Pins ===")
    # Print in rows of 6
    pins = info["pins"]
    for i in range(0, len(pins), 6):
        out.append("  " + "  ".join(f"{p:20s}" for p in pins[i:i + 6]))
    out.append("")

    if info["params"]:
        out.append("=== Params ===")
        for p in info["params"][:20]:
            out.append(f"  {p}")
        if len(info["params"]) > 20:
            out.append(f"  ... and {len(info['params']) - 20} more")
        out.append("")

    dc = stats["device_counts"]
    total = sum(dc.values())
    out.append(f"=== Direct device census (inside {name} body only) ===")
    out.append(f"Total device lines: {total:,}")
    for prefix, label in DEVICE_PREFIXES.items():
        c = dc.get(prefix, 0)
        if c > 0:
            out.append(f"  {label:22s} ({prefix}): {c:,}")
    out.append("")

    out.append("=== Unique nets referenced by body devices ===")
    out.append(f"Count: {len(stats['nets']):,}")
    # Split into pin-nets (appear in the header pin list) vs internal
    pin_set = set(info["pins"])
    pin_nets_used = sorted(n for n in stats["nets"] if n in pin_set)
    internal_nets = sorted(n for n in stats["nets"] if n not in pin_set)
    out.append(f"  pin nets appearing in body:  {len(pin_nets_used)} / {len(info['pins'])} pins")
    out.append(f"  internal nets:                {len(internal_nets):,}")
    if internal_nets:
        preview = internal_nets[:30]
        out.append(f"  first 30 internal nets: {' '.join(preview)}")
    out.append("")

    out.append("=== Top 20 directly-instantiated subckts ===")
    for sub, c in stats["subckt_instances"].most_common(20):
        out.append(f"  {sub:40s} {c:,}")
    out.append("")

    out.append("=== Top 15 models (direct) ===")
    for m, c in stats["model_counts"].most_common(15):
        out.append(f"  {m:40s} {c:,}")
    out.append("")

    out.append("=== Sample device lines (3 per type) ===")
    for prefix in ["M", "R", "C", "D", "Q", "X"]:
        samples = stats["samples"].get(prefix, [])
        if not samples:
            continue
        out.append(f"-- {DEVICE_PREFIXES[prefix]} ({prefix}) --")
        for lineno, text in samples:
            out.append(f"  L{lineno}: {text}")
    out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Dump a subckt's body from a CDL/SPICE netlist.")
    p.add_argument("netlist")
    p.add_argument("--name", required=True, help="Subckt name to extract")
    p.add_argument("--output", help="Write report to file instead of stdout")
    p.add_argument("--raw", help="Also write the raw joined body lines to this file")
    args = p.parse_args()

    if not os.path.exists(args.netlist):
        print(f"Error: file not found: {args.netlist}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for .SUBCKT {args.name} in {args.netlist} ...", file=sys.stderr)
    info = find_and_extract(args.netlist, args.name)
    if info is None:
        print(f"Error: subckt '{args.name}' not found.", file=sys.stderr)
        sys.exit(2)

    print(f"  Found at line {info['start']}, ends at {info['end']} "
          f"({info['end'] - info['start']} lines, {len(info['body'])} statements)",
          file=sys.stderr)

    stats = analyze_body(info["body"])
    text = report(args.netlist, args.name, info, stats)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)

    if args.raw:
        with open(args.raw, "w") as fh:
            for lineno, stmt in info["body"]:
                fh.write(f"{stmt}\n")
        print(f"Raw body written to {args.raw}", file=sys.stderr)


if __name__ == "__main__":
    main()
