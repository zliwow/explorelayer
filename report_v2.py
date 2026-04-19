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
    "You are an IC packaging and layout reliability reviewer. "
    "You will be given the output of a hierarchy-aware overlap detector "
    "run on a real power-management chip. Be calibrated: metal under "
    "pads is normal routing in power ICs — not every flagged finding is "
    "a real risk. Write a review in plain prose, organized by tier "
    "(high → medium → low), pointing out which findings are worth a "
    "human's time and which are geometric noise. No markdown headers, "
    "no bullets, no preamble — start with the analysis directly."
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

def build_llm_prompt(data, max_groups_per_tier=8):
    summary = data.get("summary", {})
    groups = data.get("groups", [])
    by_tier = {"high": [], "medium": [], "low": [], "info": []}
    for g in groups:
        by_tier.setdefault(g["severity_tier"], []).append(g)

    out = []
    out.append(f"Chip: {data.get('gds_file', '(unknown)')}")
    out.append(f"Top cell: {data.get('cell', '(unknown)')}")
    counts = summary.get("tier_counts_grouped") or {}
    out.append(
        "Tier distribution (groups): "
        f"high={counts.get('high', 0)} "
        f"medium={counts.get('medium', 0)} "
        f"low={counts.get('low', 0)} "
        f"info={counts.get('info', 0)}"
    )
    out.append("")

    for tier in ["high", "medium", "low"]:
        tgroups = by_tier.get(tier, [])
        if not tgroups:
            continue
        out.append(f"# {tier.upper()} findings "
                   f"(showing top {min(max_groups_per_tier, len(tgroups))} of {len(tgroups)})")
        for i, g in enumerate(tgroups[:max_groups_per_tier], 1):
            owner = g.get("owner_path") or "(unresolved)"
            reasons = ", ".join(g.get("severity_reasons") or []) or "—"
            sub = g.get("subckt_match")
            out.append(
                f"  {i}. owner={owner}  layer={g['check_layer']}  "
                f"count={g['count']}  area={g['total_overlap_area']:.1f}  "
                f"reasons={reasons}" + (f"  subckt={sub}" if sub else "")
            )
        out.append("")

    info_n = len(by_tier.get("info", []))
    if info_n:
        out.append(f"# INFO tier: {info_n} groups, not listed. These are "
                   "pad-over-routed-metal hits on cells without a sensitive "
                   "name pattern (generic routing).")
        out.append("")

    out.append(
        "Write the review now. For each tier, call out which specific "
        "findings deserve a designer's attention and which are probably "
        "fine. Be specific — quote the owner path or cell leaf name so a "
        "reader can grep the layout. End with one short paragraph on "
        "overall layout health."
    )
    return "\n".join(out)


def write_analysis_md(data, out_path, llm_url, llm_model):
    prompt = build_llm_prompt(data)
    model = llm_model or discover_model(llm_url)
    if not model:
        with open(out_path, "w") as f:
            f.write("# LLM Risk Assessment\n\n")
            f.write("_LLM server not reachable. Skipped._\n")
        print(f"  analysis.md: server not reachable -> stub written")
        return

    print(f"  Calling LLM {llm_url}  model={model} ...", end="", flush=True)
    text = call_llm(prompt, llm_url, model)
    if not text:
        print(" failed")
        with open(out_path, "w") as f:
            f.write("# LLM Risk Assessment\n\n_LLM call failed._\n")
        return

    print(" ok")
    with open(out_path, "w") as f:
        f.write(f"# LLM Risk Assessment\n\n")
        f.write(f"**Model:** {model}\n\n")
        f.write(f"**Chip:** {data.get('gds_file', '(unknown)')}\n\n")
        f.write("---\n\n")
        f.write(text)
        f.write("\n")


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
        write_analysis_md(data, analysis_path, args.llm_url, args.llm_model)
        print(f"Wrote {analysis_path}")


if __name__ == "__main__":
    main()
