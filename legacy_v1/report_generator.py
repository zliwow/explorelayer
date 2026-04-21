#!/usr/bin/env python3
"""
Report Generator — turn overlap detection JSON into human-readable
markdown reports or LLM prompt payloads for risk assessment.

Produces a CONDENSED summary grouped by pad region rather than
listing every individual polygon overlap.

Usage:
    python report_generator.py overlap_results.json --output report.md
    python report_generator.py overlap_results.json --llm-prompt > prompt.txt
"""

import argparse
import json
import sys
from collections import defaultdict


def load_results(path):
    """Load the JSON output from find_overlaps.py."""
    with open(path) as f:
        return json.load(f)


def group_findings(findings):
    """
    Group findings by region (using region_bounds as key), then by layer.
    Returns a structured summary per pad/RDL region.
    """
    regions = defaultdict(lambda: {
        "type": None,
        "bounds": None,
        "area": 0,
        "layers": defaultdict(lambda: {
            "count": 0,
            "total_overlap_area": 0,
            "max_overlap_area": 0,
            "max_overlap_pct": 0,
        })
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
        r["layers"][layer]["max_overlap_area"] = max(
            r["layers"][layer]["max_overlap_area"], f["overlap_area"])
        r["layers"][layer]["max_overlap_pct"] = max(
            r["layers"][layer]["max_overlap_pct"], f["overlap_pct_of_region"])

    return dict(regions)


def risk_score(region_summary):
    """
    Assign a risk level to a pad region based on what's under it.
    Returns (level, score, reasons).

    Scoring philosophy:
      - Poly / active / diffusion under pad = real risk (gate oxide stress,
        reflow shift of precision devices, cracking).
      - Metal routing under pad = NORMAL in power IC layout. We record it
        for context but do NOT score it as risk — otherwise every densely
        routed pad trips HIGH which is a false alarm.
    """
    reasons = []
    score = 0
    metal_layers_present = 0
    total_metal_overlaps = 0

    for layer, stats in region_summary["layers"].items():
        lname = layer.lower()

        # Poly under pad — reflow shift, gate-oxide stress cracking
        if "poly" in lname:
            score += 40
            reasons.append(f"POLY under pad: {stats['count']} structures, "
                           f"{stats['total_overlap_area']:.1f} area, "
                           f"max {stats['max_overlap_pct']:.0f}% of pad")

        # Active / diffusion / OD — precision device under pad
        elif "active" in lname or "diff" in lname or lname.startswith("od"):
            score += 30
            reasons.append(f"ACTIVE under pad: {stats['count']} structures, "
                           f"max {stats['max_overlap_pct']:.0f}% of pad")

        # High-resistance poly resistors — these are the precision analog devices
        elif "rpoly" in lname or "hires" in lname or "resistor" in lname:
            score += 35
            reasons.append(f"PRECISION RESISTOR under pad: {stats['count']} structures")

        # Metal routing — NORMAL, count but don't score
        elif "metal" in lname:
            metal_layers_present += 1
            total_metal_overlaps += stats["count"]

    # Append metal summary as informational (not scored)
    if metal_layers_present > 0:
        reasons.append(
            f"[info] Metal routing: {metal_layers_present} layers, "
            f"{total_metal_overlaps} total overlaps — normal, not scored"
        )

    if score >= 30:
        level = "HIGH"
    elif score >= 15:
        level = "MEDIUM"
    elif score > 0:
        level = "LOW"
    else:
        level = "CLEAR"

    return level, score, reasons


# ── Markdown report ──────────────────────────────────────────────────────────

def generate_markdown(data):
    """Generate a condensed markdown report grouped by pad region."""
    lines = []
    lines.append(f"# GDS Overlap Report — {data['gds_file']}")
    lines.append(f"**Cell:** {data['cell']}\n")

    summary = data["summary"]
    lines.append("## Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Pad regions analysed | {summary['total_pad_regions']} |")
    lines.append(f"| RDL regions analysed | {summary['total_rdl_regions']} |")
    lines.append(f"| Total overlaps found | {summary['total_overlaps']} |")
    lines.append("")

    cfg = data["config"]
    lines.append("## Layer Configuration\n")
    lines.append(f"- **Pad layers:** {cfg['pad_layers']}")
    lines.append(f"- **RDL layers:** {cfg['rdl_layers']}")
    lines.append(f"- **Check layers:**")
    for cl in cfg["check_layers"]:
        lines.append(f"  - {cl['name']} (layer {cl['layer']})")
    lines.append(f"- **Min overlap area:** {cfg['min_overlap_area']}")
    lines.append("")

    if not data["findings"]:
        lines.append("## Findings\n")
        lines.append("No overlaps detected.\n")
        return "\n".join(lines)

    # Group and score
    grouped = group_findings(data["findings"])

    # Separate by region type
    pad_regions = {k: v for k, v in grouped.items() if v["type"] == "pad"}
    rdl_regions = {k: v for k, v in grouped.items() if v["type"] == "rdl"}

    # Score and sort pads by risk
    scored_pads = []
    for key, region in pad_regions.items():
        level, score, reasons = risk_score(region)
        scored_pads.append((key, region, level, score, reasons))
    scored_pads.sort(key=lambda x: -x[3])

    # Risk overview
    risk_counts = defaultdict(int)
    for _, _, level, _, _ in scored_pads:
        risk_counts[level] += 1

    lines.append("## Risk Overview\n")
    lines.append(f"| Risk Level | Pad Count |")
    lines.append(f"|------------|-----------|")
    for level in ["HIGH", "MEDIUM", "LOW", "CLEAR"]:
        if risk_counts[level] > 0:
            lines.append(f"| {level} | {risk_counts[level]} |")
    lines.append("")

    # Detailed per-pad report (HIGH and MEDIUM only in detail, LOW/CLEAR summarized)
    lines.append("## Pad Region Details\n")

    for key, region, level, score, reasons in scored_pads:
        if level in ("HIGH", "MEDIUM"):
            b = region["bounds"]
            lines.append(f"### Pad at ({b[0]:.0f}, {b[1]:.0f})..({b[2]:.0f}, {b[3]:.0f}) — **{level} RISK**\n")
            lines.append(f"Pad area: {region['area']:.1f}\n")
            lines.append("**Risk factors:**")
            for r in reasons:
                lines.append(f"- {r}")
            lines.append("")
            lines.append("| Layer | Overlapping Geometries | Total Overlap Area | Max Single Overlap % |")
            lines.append("|-------|----------------------|-------------------|---------------------|")
            for layer, stats in sorted(region["layers"].items()):
                lines.append(f"| {layer} | {stats['count']} | {stats['total_overlap_area']:.1f} "
                             f"| {stats['max_overlap_pct']:.1f}% |")
            lines.append("")

    # Summary table for LOW/CLEAR pads
    low_clear = [(k, r, l, s, rs) for k, r, l, s, rs in scored_pads if l in ("LOW", "CLEAR")]
    if low_clear:
        lines.append("### Low Risk / Clear Pads (summary)\n")
        lines.append("| Pad Location | Risk | Layers Present | Total Overlaps |")
        lines.append("|-------------|------|----------------|----------------|")
        for key, region, level, score, reasons in low_clear:
            b = region["bounds"]
            layer_names = ", ".join(sorted(region["layers"].keys()))
            total_overlaps = sum(s["count"] for s in region["layers"].values())
            lines.append(f"| ({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f}) "
                         f"| {level} | {layer_names} | {total_overlaps} |")
        lines.append("")

    # RDL section (condensed)
    if rdl_regions:
        lines.append("## RDL Region Summary\n")
        lines.append(f"**{len(rdl_regions)} RDL regions analyzed.**\n")
        # Just aggregate stats
        rdl_layer_totals = defaultdict(lambda: {"regions_affected": 0, "total_overlaps": 0})
        for region in rdl_regions.values():
            for layer, stats in region["layers"].items():
                rdl_layer_totals[layer]["regions_affected"] += 1
                rdl_layer_totals[layer]["total_overlaps"] += stats["count"]
        lines.append("| Check Layer | RDL Regions Affected | Total Overlaps |")
        lines.append("|-------------|---------------------|----------------|")
        for layer, totals in sorted(rdl_layer_totals.items()):
            lines.append(f"| {layer} | {totals['regions_affected']} | {totals['total_overlaps']} |")
        lines.append("")

    lines.append("---\n*Generated by gds-layout-explorer*\n")
    return "\n".join(lines)


# ── LLM prompt payload ──────────────────────────────────────────────────────

def generate_llm_prompt(data):
    """
    Generate a condensed prompt for an LLM to assess risk.
    Only includes HIGH and MEDIUM risk findings.
    """
    parts = []

    parts.append("You are an IC packaging and layout reliability expert. "
                  "Analyze the following overlap findings from a GDSII layout "
                  "and assess the risk each one poses.\n")

    parts.append("## Context\n")
    parts.append(f"- **Chip:** {data['gds_file']} (cell: {data['cell']})")
    parts.append(f"- **Total pad regions:** {data['summary']['total_pad_regions']}")
    parts.append(f"- **Total overlaps found:** {data['summary']['total_overlaps']}")
    parts.append("")

    parts.append("## Known IC Packaging Failure Modes\n")
    parts.append("- **Solder reflow shift:** Structures under pads shift ~1-5% during reflow. Poly resistors especially sensitive.")
    parts.append("- **Stress cracking:** Mechanical stress from bonding can crack underlying dielectrics or poly.")
    parts.append("- **Electromigration:** High current paths under pads accelerate EM failures.")
    parts.append("- **Oxide cracking:** Probe marks / wire bonding damage propagates to underlying structures.")
    parts.append("- **Thermal mismatch:** CTE differences create stress concentrations.")
    parts.append("")

    # Group and score
    grouped = group_findings(data["findings"])
    pad_regions = {k: v for k, v in grouped.items() if v["type"] == "pad"}

    scored = []
    for key, region in pad_regions.items():
        level, score, reasons = risk_score(region)
        scored.append((key, region, level, score, reasons))
    scored.sort(key=lambda x: -x[3])

    # Only include concerning pads
    concerning = [(k, r, l, s, rs) for k, r, l, s, rs in scored if l in ("HIGH", "MEDIUM")]

    if not concerning:
        parts.append("## Findings\n")
        parts.append("No high or medium risk overlaps detected. All pads appear clear of sensitive structures.")
    else:
        parts.append(f"## Findings ({len(concerning)} concerning pad regions)\n")
        for i, (key, region, level, score, reasons) in enumerate(concerning, 1):
            b = region["bounds"]
            parts.append(f"### Pad {i} at ({b[0]:.0f},{b[1]:.0f})..({b[2]:.0f},{b[3]:.0f}) — {level} RISK\n")
            parts.append(f"Pad area: {region['area']:.1f}")
            parts.append("Structures found under this pad:")
            for layer, stats in sorted(region["layers"].items()):
                parts.append(f"  - {layer}: {stats['count']} geometries, "
                             f"total overlap area {stats['total_overlap_area']:.1f}, "
                             f"max single overlap {stats['max_overlap_pct']:.1f}% of pad")
            parts.append("")

    parts.append("## Your Task\n")
    parts.append("For each flagged pad:")
    parts.append("1. Assign a **risk level**: HIGH, MEDIUM, or LOW")
    parts.append("2. Explain **why** — reference specific failure modes")
    parts.append("3. Provide a **recommendation** (move structure, add keep-out, acceptable, needs review)")
    parts.append("")
    parts.append("Provide an **overall assessment** and general recommendations.")

    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from GDS overlap detection results."
    )
    parser.add_argument("results_file", help="Path to the JSON results from find_overlaps.py")
    parser.add_argument("--output", default=None, help="Write markdown report to this file")
    parser.add_argument("--llm-prompt", action="store_true",
                        help="Print LLM risk-assessment prompt to stdout")
    args = parser.parse_args()

    data = load_results(args.results_file)

    if not args.output and not args.llm_prompt:
        print("Error: specify --output for markdown report and/or --llm-prompt for LLM payload.")
        sys.exit(1)

    if args.output:
        md = generate_markdown(data)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Markdown report written to {args.output}", file=sys.stderr)

    if args.llm_prompt:
        prompt = generate_llm_prompt(data)
        print(prompt)


if __name__ == "__main__":
    main()
