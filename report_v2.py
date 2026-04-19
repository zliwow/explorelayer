#!/usr/bin/env python3
"""
report_v2.py — turn findings_v2.json into two readable markdowns.

Writes report.md (detector summary, tiered risk, top groups per tier) and
analysis.md (Qwen narrative — optional, skipped if --no-llm).

Usage:
    python report_v2.py findings_v2.json
    python report_v2.py findings_v2.json --output-dir reports/
    python report_v2.py findings_v2.json --no-llm
    python report_v2.py findings_v2.json --llm-url http://localhost:8000
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


TIER_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}
TOP_N_PER_TIER = 10


# ── LLM plumbing (Qwen / OpenAI-compatible) ─────────────────────────────

LLM_SYSTEM = (
    "You are an IC packaging and layout reliability reviewer. You will "
    "be given ONE finding at a time from a hierarchy-aware overlap "
    "detector run on a real power-management chip. Be calibrated: metal "
    "routing under pads is normal in power ICs — not every flagged "
    "overlap is a real risk. In 2-3 short sentences, say (a) whether "
    "this is a real reliability risk or likely geometric noise, (b) the "
    "physical mechanism if it IS a risk, and (c) one concrete thing the "
    "designer should check. No preamble, no headers, no bullets — start "
    "with the assessment directly."
)


def discover_model(base_url, timeout=5):
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        return data[0].get("id") if data else None
    except Exception:
        return None


def call_llm(prompt, base_url, model, timeout=600):
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        # strip <think>, markdown boldening, header prefixes
        text = re.sub(r"<think>.*?</think>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError,
            KeyError, TypeError, AttributeError,
            json.JSONDecodeError, TimeoutError) as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        return None


# ── Report rendering ────────────────────────────────────────────────────

def tier_counts_table(summary):
    counts = summary.get("tier_counts_grouped") or {}
    rows = ["| Tier | Groups |", "|------|-------:|"]
    for tier in ["high", "medium", "low", "info"]:
        rows.append(f"| {tier} | {counts.get(tier, 0)} |")
    return "\n".join(rows)


def group_line(g, idx):
    owner = g.get("owner_path") or "(unresolved)"
    leaf = owner.rsplit("/", 1)[-1] if "/" in owner else owner
    reasons = ", ".join(g.get("severity_reasons") or []) or "—"
    subckt = g.get("subckt_match")
    subckt_s = f"  \nSubckt match: `{subckt}`" if subckt else ""
    return (
        f"### {idx}. `{leaf}` — {g['check_layer']} "
        f"[{g['severity_tier']}]\n\n"
        f"- Owner path: `{owner}`\n"
        f"- Overlap count: {g['count']}\n"
        f"- Total overlap area: {g['total_overlap_area']:.1f}\n"
        f"- Severity reasons: {reasons}"
        f"{subckt_s}\n"
    )


def write_report_md(data, out_path):
    summary = data.get("summary", {})
    groups = data.get("groups", [])

    lines = []
    lines.append(f"# Overlap detector v2 — {data.get('gds_file', '(unknown)')}\n")
    lines.append(f"Top cell: `{data.get('cell', '(unknown)')}`  ")
    if data.get("netlist"):
        lines.append(f"Netlist: `{data['netlist']}`  ")
    lines.append(f"Total raw findings: {summary.get('total_raw_findings', 0)}  ")
    lines.append(f"Total groups (owner × check_layer): "
                 f"{summary.get('total_groups', 0)}\n")

    lines.append("## Risk summary\n")
    lines.append(tier_counts_table(summary) + "\n")

    by_tier = {"high": [], "medium": [], "low": [], "info": []}
    for g in groups:
        by_tier.setdefault(g["severity_tier"], []).append(g)

    for tier in ["high", "medium", "low"]:
        tgroups = by_tier.get(tier, [])
        if not tgroups:
            continue
        lines.append(f"## {tier.upper()} severity — "
                     f"{len(tgroups)} group(s), top {min(TOP_N_PER_TIER, len(tgroups))}\n")
        for i, g in enumerate(tgroups[:TOP_N_PER_TIER], 1):
            lines.append(group_line(g, i))

    info_count = len(by_tier.get("info", []))
    if info_count:
        lines.append(f"## INFO tier (noise) — {info_count} group(s), not listed\n")
        lines.append("These are routed metal under pads on non-sensitive "
                     "cells — expected in a power IC layout.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ── LLM prompt ──────────────────────────────────────────────────────────

def build_finding_prompt(g, chip_name):
    owner = g.get("owner_path") or "(unresolved)"
    reasons = ", ".join(g.get("severity_reasons") or []) or "—"
    sub = g.get("subckt_match")
    lines = [
        f"Chip: {chip_name}",
        f"Severity tier: {g['severity_tier']}",
        f"Owner path: {owner}",
        f"Check layer: {g['check_layer']}",
        f"Overlap count: {g['count']}",
        f"Total overlap area: {g['total_overlap_area']:.1f}",
        f"Severity reasons (pattern matches): {reasons}",
    ]
    if sub:
        lines.append(f"Subckt match in netlist: {sub}")
    lines.append("")
    lines.append("Write the 2-3 sentence assessment now.")
    return "\n".join(lines)


def group_header(i, g):
    owner = g.get("owner_path") or "(unresolved)"
    leaf = owner.rsplit("/", 1)[-1] if "/" in owner else owner
    return (f"### {g['severity_tier'].upper()} #{i}: `{leaf}` — "
            f"{g['check_layer']}  (count={g['count']}, "
            f"area={g['total_overlap_area']:.1f})")


def write_analysis_md(data, out_path, llm_url, llm_model, max_findings):
    groups = data.get("groups", [])
    # Keep only tiers we actually review, in order
    tiered = []
    for tier in ["high", "medium", "low"]:
        for g in groups:
            if g["severity_tier"] == tier:
                tiered.append(g)
    if max_findings and len(tiered) > max_findings:
        print(f"  capping LLM pass at {max_findings} findings "
              f"(of {len(tiered)} total non-info groups)")
        tiered = tiered[:max_findings]

    model = llm_model or discover_model(llm_url)
    chip_name = data.get("gds_file", "(unknown)")

    # Open output file and write incrementally — if the pass dies halfway
    # you still have partial analysis on disk.
    with open(out_path, "w") as f:
        f.write("# LLM Risk Assessment — per-finding\n\n")
        f.write(f"**Chip:** {chip_name}\n\n")
        if not model:
            f.write("_LLM server not reachable — no per-finding analysis._\n")
            print("  analysis.md: server not reachable -> stub written")
            return
        f.write(f"**Model:** {model}\n\n")
        f.write(f"**Findings reviewed:** {len(tiered)} "
                f"(high → medium → low)\n\n")
        f.write("---\n\n")
        f.flush()

        current_tier = None
        for i, g in enumerate(tiered, 1):
            if g["severity_tier"] != current_tier:
                current_tier = g["severity_tier"]
                f.write(f"\n## {current_tier.upper()} tier\n\n")
            f.write(group_header(i, g) + "\n\n")

            prompt = build_finding_prompt(g, chip_name)
            print(f"  [{i}/{len(tiered)}] {current_tier:6s} "
                  f"{(g.get('owner_path') or '')[-70:]} ...",
                  end="", flush=True)
            text = call_llm(prompt, llm_url, model)
            if text:
                print(" ok")
                f.write(text + "\n\n")
            else:
                print(" failed")
                f.write("_LLM call failed for this finding._\n\n")
            f.flush()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("findings_json")
    p.add_argument("--output-dir", default=None,
                   help="Default: same dir as findings_json.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip analysis.md (detector-only report).")
    p.add_argument("--llm-url", default="http://localhost:8000")
    p.add_argument("--llm-model", default=None,
                   help="Default: auto-discover from /v1/models.")
    p.add_argument("--max-findings", type=int, default=40,
                   help="Cap per-finding LLM calls. Default 40.")
    args = p.parse_args()

    with open(args.findings_json) as f:
        data = json.load(f)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.findings_json))
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, "report.md")
    write_report_md(data, report_path)
    print(f"Wrote {report_path}")

    if not args.no_llm:
        analysis_path = os.path.join(out_dir, "analysis.md")
        write_analysis_md(data, analysis_path, args.llm_url, args.llm_model,
                          args.max_findings)
        print(f"Wrote {analysis_path}")


if __name__ == "__main__":
    main()
