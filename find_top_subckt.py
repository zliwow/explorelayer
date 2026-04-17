#!/usr/bin/env python3
"""
Find the chip-top subckt in a CDL/SPICE netlist.

Lists every .SUBCKT definition, computes which subckts are roots
(defined but never instantiated via X-lines), and ranks candidates
by pin count. Optionally matches a known GDS top-cell name.

Usage:
    python find_top_subckt.py MPQ8897_ESD_R4.cdl
    python find_top_subckt.py MPQ8897_ESD_R4.cdl --gds-top mpq8897r4_1
    python find_top_subckt.py MPQ8897_ESD_R4.cdl --gds-top mpq8897r4_1 --output top_report.txt
"""

import argparse
import os
import re
import sys
from collections import Counter

PARAM_RE = re.compile(r"^\w+\s*=")


def scan(path):
    defs = []          # list of (name, pin_count, line_no, pins_sample)
    x_refs = Counter() # subckt name -> instance count

    with open(path, "r", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            first = stripped[0]
            if first in ("*", "$") or stripped.startswith("//"):
                continue
            if first == "+":
                continue

            low = stripped.lower()

            if low.startswith(".subckt") or low.startswith("subckt "):
                parts = stripped.split()
                if len(parts) >= 2:
                    name = parts[1]
                    pins = [p for p in parts[2:] if "=" not in p]
                    defs.append((name, len(pins), lineno, pins[:20]))
                continue

            if first.upper() == "X":
                tokens = stripped.split()
                non_param = [t for t in tokens[1:] if not PARAM_RE.match(t)]
                if non_param:
                    x_refs[non_param[-1]] += 1

    return defs, x_refs


def rank_candidates(defs, x_refs, gds_top=None):
    defined_names = {d[0] for d in defs}
    used_names = set(x_refs.keys())
    roots = defined_names - used_names

    by_name = {d[0]: d for d in defs}
    root_info = [by_name[n] for n in roots]
    root_info.sort(key=lambda d: d[1], reverse=True)

    gds_match_exact = None
    gds_match_fuzzy = []
    if gds_top:
        gt = gds_top.lower()
        # Token-based fuzzy: split on _ and non-alnum, compare token overlap
        gt_tokens = set(re.split(r"[^a-z0-9]+", gt)) - {""}
        for name, pins, ln, ps in defs:
            if name.lower() == gt:
                gds_match_exact = (name, pins, ln, ps)
        scored = []
        for name, pins, ln, ps in defs:
            if gds_match_exact and name == gds_match_exact[0]:
                continue
            n_tokens = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
            overlap = len(gt_tokens & n_tokens)
            if overlap == 0:
                continue
            scored.append((overlap, pins, name, ln, ps))
        scored.sort(reverse=True)  # highest overlap, then pin count
        gds_match_fuzzy = [(n, p, l, ps) for _, p, n, l, ps in scored[:10]]

    return root_info, gds_match_exact, gds_match_fuzzy


def report(path, defs, x_refs, root_info, gds_match_exact, gds_match_fuzzy, gds_top):
    out = []
    out.append(f"=== Top-subckt finder: {os.path.basename(path)} ===")
    out.append(f"Total subckts defined:        {len(defs):,}")
    out.append(f"Unique subckts X-instantiated: {len(x_refs):,}")
    out.append(f"Root subckts (defined, never instantiated): {len(root_info):,}")
    out.append("")

    if gds_top:
        out.append(f"=== Match against GDS top cell '{gds_top}' ===")
        if gds_match_exact:
            name, pins, ln, ps = gds_match_exact
            out.append(f"  EXACT MATCH: {name}  ({pins} pins, line {ln})")
            out.append(f"  first pins: {' '.join(ps)}")
        else:
            out.append("  No exact match.")
        if gds_match_fuzzy:
            out.append("  Fuzzy matches (substring):")
            for name, pins, ln, _ in gds_match_fuzzy[:10]:
                out.append(f"    {name:40s} {pins:4d} pins  L{ln}")
        out.append("")

    out.append("=== Top 20 root subckts by pin count (best chip-top candidates) ===")
    for name, pins, ln, ps in root_info[:20]:
        out.append(f"  {name:40s} {pins:4d} pins  L{ln}")
    out.append("")

    out.append("=== Top 20 ALL subckts by pin count (regardless of root) ===")
    all_sorted = sorted(defs, key=lambda d: d[1], reverse=True)
    for name, pins, ln, ps in all_sorted[:20]:
        mark = "(root)" if name in {r[0] for r in root_info} else ""
        out.append(f"  {name:40s} {pins:4d} pins  L{ln}  {mark}")
    out.append("")

    out.append("=== Recommendation ===")
    chosen = None
    root_names = {r[0] for r in root_info}
    if gds_match_exact:
        chosen = gds_match_exact
        out.append(f"  Use: {chosen[0]}  (exact match to GDS top)")
    elif gds_match_fuzzy:
        # Prefer a fuzzy match that is ALSO a root subckt
        for cand in gds_match_fuzzy:
            if cand[0] in root_names:
                chosen = cand
                out.append(f"  Use: {chosen[0]}  (token-match to GDS top AND a root)")
                break
        if not chosen:
            chosen = gds_match_fuzzy[0]
            out.append(f"  Use: {chosen[0]}  (best token-match to GDS top)")
    elif root_info:
        chosen = root_info[0]
        out.append(f"  Use: {chosen[0]}  ({chosen[1]} pins — largest root subckt)")
        out.append("  NOTE: verify by eyeballing the pin list below; a real chip top")
        out.append("  usually has power rails (VDD/VSS/...) and many signal pins.")
    if chosen:
        out.append(f"  Pins (first 20): {' '.join(chosen[3])}")
    out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Find chip-top subckt in a CDL/SPICE netlist.")
    p.add_argument("netlist")
    p.add_argument("--gds-top", help="Known GDS top-cell name (optional)")
    p.add_argument("--output", help="Write report to file instead of stdout")
    args = p.parse_args()

    if not os.path.exists(args.netlist):
        print(f"Error: file not found: {args.netlist}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.netlist} ...", file=sys.stderr)
    defs, x_refs = scan(args.netlist)
    root_info, gme, gmf = rank_candidates(defs, x_refs, args.gds_top)
    text = report(args.netlist, defs, x_refs, root_info, gme, gmf, args.gds_top)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
