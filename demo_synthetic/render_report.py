#!/usr/bin/env python3
"""
Render a PPT-slide-style report from findings.json:

  +----------------------------+  +----------------------------+
  |                            |  |  BANDGAP_REF subckt:       |
  |   [GDS layout view]        |  |    XQ1 ... BJT_PNP         |
  |   red circle(s) on issues  |  |    XQ2 ... BJT_PNP         |
  |                            |  |    (matched pair)          |
  +----------------------------+  +----------------------------+
  Caption: BJT diff pair under RDL — Claude finding.

Outputs demo_report.png.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

import gdstk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D


# ── LLM backend (OpenAI-compatible chat completions) ────────────────────
# Default points at the work-machine Qwen/sglang server at localhost:8000.
# If the request fails we fall back to the original hardcoded narrative
# below so the demo still renders end-to-end.

LLM_SYSTEM = (
    "You are an analog/mixed-signal layout reviewer. Given a detector "
    "report about RDL (redistribution-layer metal) crossing a matched "
    "BJT pair in a bandgap reference, write a review in the form of 4-5 "
    "short paragraphs separated by a blank line. "
    "Paragraph 1: identify the issue (which devices, which block, how much "
    "overlap). "
    "Paragraph 2-4: the three distinct physical mechanisms that make this "
    "a real problem (thermal asymmetry, piezo-resistive stress, capacitive "
    "coupling) — one paragraph each, with specific quantitative cues. "
    "Paragraph 5: recommended fix (reroute, shield, rerun MC). "
    "Plain prose only: no headers, no markdown, no bullet points, no lists, "
    "no thinking, no preamble. Start with paragraph 1 directly."
)


def _clean_text(text):
    if not text:
        return text
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _discover_model(llm_url, timeout=5):
    if "/chat/completions" in llm_url:
        url = llm_url.replace("/chat/completions", "/models")
    else:
        url = llm_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        if data:
            return data[0].get("id")
    except Exception:
        return None
    return None


def _call_llm(prompt, url, model, timeout=600):
    # Thinking is ON by design: the whole point of this demo is reasoning
    # depth. Qwen3 writes its chain-of-thought to reasoning_content (or a
    # <think> block) and the final answer to content. We give it a large
    # token budget so thinking + answer both fit; small budgets caused
    # content=null earlier because thinking consumed all tokens.
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return _clean_text(text) or None
    except (urllib.error.URLError, urllib.error.HTTPError,
            KeyError, TypeError, AttributeError,
            json.JSONDecodeError, TimeoutError) as exc:
        print(f"  LLM call failed: {exc}", file=sys.stderr)
        return None


# Layer -> (color, alpha, zorder, label) for rendering
STYLE = {
    (1, 0):   ("#2e7d32", 0.6, 1, "diff"),
    (5, 0):   ("#c62828", 0.6, 2, "poly"),
    (10, 0):  ("#37474f", 0.8, 3, "contact"),
    (11, 0):  ("#1976d2", 0.45, 4, "met1"),
    (22, 0):  ("#8e24aa", 0.45, 5, "met2"),
    (33, 0):  ("#ef6c00", 0.35, 6, "met3"),
    (33, 1):  ("#aaaaaa", 0.0, 0, "outline"),  # boundary-only
    (50, 0):  ("#ffeb3b", 0.35, 8, "RDL"),
    (60, 0):  ("#000000", 0.9, 7, "pad_open"),
    (82, 5):  ("#ad1457", 0.4, 6, "BJT"),
}


def flatten_polygons(cell, origin=(0.0, 0.0), by_layer=None):
    """Return {(layer, dt): [polys_points]} flattened in top-cell coordinates."""
    if by_layer is None:
        by_layer = {}
    for p in cell.polygons:
        key = (p.layer, p.datatype)
        shifted = [(pt[0] + origin[0], pt[1] + origin[1]) for pt in p.points]
        by_layer.setdefault(key, []).append(shifted)
    for ref in cell.references:
        child = ref.cell
        if isinstance(child, gdstk.Cell):
            new_origin = (origin[0] + float(ref.origin[0]),
                          origin[1] + float(ref.origin[1]))
            flatten_polygons(child, new_origin, by_layer)
    return by_layer


def render_layout(ax, top):
    by_layer = flatten_polygons(top)
    for (layer, dt), polys in sorted(by_layer.items(),
                                     key=lambda kv: STYLE.get(kv[0], ("", 0, 0, ""))[2]):
        style = STYLE.get((layer, dt))
        if style is None:
            continue
        color, alpha, _, _ = style
        for pts in polys:
            if alpha == 0.0:  # outline only
                mp = MplPolygon(pts, closed=True, fill=False,
                                edgecolor=color, linewidth=0.6, alpha=0.5)
            else:
                mp = MplPolygon(pts, closed=True, facecolor=color, alpha=alpha,
                                edgecolor=color, linewidth=0.4)
            ax.add_patch(mp)

    # Labels
    for lbl in top.labels:
        x, y = float(lbl.origin[0]), float(lbl.origin[1])
        ax.text(x, y, lbl.text, fontsize=5, color="white",
                ha="center", va="center",
                bbox=dict(facecolor="#000000", alpha=0.55, pad=0.8,
                          edgecolor="none"))

    # Die bounds
    minx, miny, maxx, maxy = get_bounds(top)
    pad = 15
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    ax.set_title(f"GDS: {top.name}  —  layout view", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")


def get_bounds(cell):
    bbox = cell.bounding_box()
    if bbox is None:
        return 0, 0, 100, 100
    (minx, miny), (maxx, maxy) = bbox
    return minx, miny, maxx, maxy


def draw_issue_circles(ax, findings_report):
    """Draw big red circle + label for every critical/high group."""
    annotated_positions = []
    for group in findings_report["groups"]:
        sev = group["sensitivity"]
        if sev not in ("critical", "high"):
            continue
        for item in group["items"]:
            minx, miny, maxx, maxy = item["overlap_bounds"]
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            r = max(maxx - minx, maxy - miny) / 2 + 6
            color = "#ff1744" if sev == "critical" else "#ff9100"
            circ = patches.Circle((cx, cy), radius=r, fill=False,
                                  edgecolor=color, linewidth=2.2, zorder=20)
            ax.add_patch(circ)
            annotated_positions.append((cx, cy, sev, item["instance_path"],
                                        group["check_layer"]))

    # Legend marker
    if annotated_positions:
        legend_items = [
            Line2D([0], [0], marker="o", color="#ff1744", linewidth=0,
                   markerfacecolor="none", markeredgewidth=2, markersize=10,
                   label="CRITICAL issue"),
            Line2D([0], [0], marker="o", color="#ff9100", linewidth=0,
                   markerfacecolor="none", markeredgewidth=2, markersize=10,
                   label="HIGH-severity"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=8,
                  facecolor="#222222", edgecolor="white", labelcolor="white")


def build_finding_text(report, netlist_path):
    """Compose the right-panel prose / subckt extract."""
    # Pick the top critical group
    crit = [g for g in report["groups"] if g["sensitivity"] == "critical"]
    if not crit:
        return "No critical findings."

    # Extract device instances grouped by path parent
    by_parent = {}
    for g in crit:
        parent = g["instance_path"].split(" / ")[-2] if " / " in g["instance_path"] else "?"
        leaf = g["instance_path"].split(" / ")[-1]
        by_parent.setdefault(parent, []).append((leaf, g))

    # Load the subckt body so we can quote it
    subckt_body = extract_subckt_body(netlist_path, "BANDGAP_REF")

    lines = []
    lines.append("FINDING  —  RDL over matched BJT pair")
    lines.append("=" * 42)
    lines.append("")
    for parent, groups in by_parent.items():
        lines.append(f"Block:  {parent}")
        instances = sorted({g.split(" / ")[-1].split("@")[0] + "@" + g.split("@")[-1]
                             for _, g_obj in groups for g in [g_obj['instance_path']]})
        lines.append(f"Devices under RDL: {len(groups)} {groups[0][0].split('@')[0]} instances")
        for (leaf, g) in groups:
            coords = leaf.split("@")[-1] if "@" in leaf else ""
            lines.append(f"   - {leaf.split('@')[0]} at {coords}   "
                         f"overlap={g['total_overlap_area']:.0f} um^2")
        lines.append("")

    lines.append("Netlist evidence (BANDGAP_REF subckt):")
    lines.append("-" * 42)
    for bl in subckt_body[:12]:
        lines.append(f"  {bl}")
    lines.append("")
    lines.append("Why this matters:")
    lines.append("-" * 42)
    lines.append("Q1 and Q2 form the matched PNP pair that")
    lines.append("generates the PTAT voltage (ΔVbe). Their Vbe")
    lines.append("match depends on thermal & mechanical symmetry.")
    lines.append("An RDL trace across both devices introduces:")
    lines.append("  - piezo-stress asymmetry (different strain")
    lines.append("    under edge vs. centre of the RDL)")
    lines.append("  - localised IR heating if the RDL carries")
    lines.append("    current → Vbe drift, bandgap offset")
    lines.append("  - capacitive coupling into high-Z base node")
    lines.append("")
    lines.append("Recommendation: reroute RDL off the BGR block,")
    lines.append("or add an M-top shield tied to a quiet ground.")
    return "\n".join(lines)


def extract_subckt_body(netlist_path, subckt_name):
    """Return the lines of the named subckt (stripped of leading/trailing whitespace)."""
    if not os.path.exists(netlist_path):
        return [f"(netlist {netlist_path} not found)"]
    body = []
    inside = False
    with open(netlist_path) as f:
        for line in f:
            s = line.rstrip()
            sl = s.strip().lower()
            if not inside and sl.startswith(".subckt") and s.strip().split()[1].lower() == subckt_name.lower():
                inside = True
                body.append(s.strip())
                continue
            if inside:
                if sl.startswith(".ends"):
                    body.append(s.strip())
                    break
                body.append(s.strip())
    return body


def render(gds_path, cfg_path, findings_path, netlist_path, out_path,
           llm_url, llm_model):
    print(f"Loading {gds_path} ...")
    lib = gdstk.read_gds(gds_path)
    top = lib.top_level()[0]

    with open(findings_path) as f:
        report = json.load(f)

    analysis_text, llm_used = build_ai_analysis(report, llm_url, llm_model)

    # Bigger figure: 2x2 grid
    #   top-left : full layout     top-right: AI analysis (prose)
    #   bot-left : zoomed-in issue bot-right: netlist evidence
    fig = plt.figure(figsize=(20, 11), facecolor="#0e0e0e")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.4],
                           height_ratios=[1.1, 1.0],
                           wspace=0.08, hspace=0.15)

    # -- full layout view --
    ax_full = fig.add_subplot(gs[0, 0])
    render_layout(ax_full, top)
    draw_issue_circles(ax_full, report)

    # -- zoomed-in view on the critical issue --
    ax_zoom = fig.add_subplot(gs[1, 0])
    render_layout(ax_zoom, top)
    draw_issue_circles(ax_zoom, report)
    issue_bounds = compute_issue_bounds(report)
    if issue_bounds:
        minx, miny, maxx, maxy = issue_bounds
        pad = max((maxx - minx), (maxy - miny)) * 0.4 + 10
        ax_zoom.set_xlim(minx - pad, maxx + pad)
        ax_zoom.set_ylim(miny - pad, maxy + pad)
    ax_zoom.set_title("Zoom — matched BJT pair under RDL strip",
                      color="#ff5252", fontsize=11)
    # Outline on full view showing where the zoom is
    if issue_bounds:
        minx, miny, maxx, maxy = issue_bounds
        pad = 8
        ax_full.add_patch(patches.Rectangle(
            (minx - pad, miny - pad),
            (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad,
            fill=False, edgecolor="#00e5ff", linewidth=1.2,
            linestyle="--", zorder=25))

    # -- AI analysis prose (top right) --
    ax_ai = fig.add_subplot(gs[0, 1])
    ax_ai.axis("off")
    ax_ai.set_facecolor("#0e0e0e")
    ax_ai.set_xlim(0, 1); ax_ai.set_ylim(0, 1)
    ai_title = "AI ANALYSIS — Qwen" if llm_used else "AI ANALYSIS — template fallback"
    ax_ai.text(0.0, 1.0, ai_title,
               fontsize=13, weight="bold", color="#00e5ff",
               va="top", transform=ax_ai.transAxes)
    ax_ai.text(0.0, 0.93, analysis_text,
               family="DejaVu Sans", fontsize=8.5,
               verticalalignment="top", color="white",
               transform=ax_ai.transAxes, clip_on=True)

    # -- netlist evidence panel (bottom right) --
    ax_nl = fig.add_subplot(gs[1, 1])
    ax_nl.axis("off")
    ax_nl.set_facecolor("#0e0e0e")
    ax_nl.set_xlim(0, 1); ax_nl.set_ylim(0, 1)
    ax_nl.text(0.0, 1.0, "Netlist evidence — BANDGAP_REF",
               fontsize=12, weight="bold", color="#ffeb3b",
               va="top", transform=ax_nl.transAxes)
    body = extract_subckt_body(netlist_path, "BANDGAP_REF")
    highlighted = highlight_subckt(body)
    ax_nl.text(0.0, 0.90, highlighted,
               family="monospace", fontsize=9.0,
               verticalalignment="top", color="white",
               transform=ax_nl.transAxes, clip_on=True)

    crit_count = sum(1 for g in report["groups"] if g["sensitivity"] == "critical")
    fig.suptitle(
        f"AI check layout under RDL  —  {report['gds_file']}  "
        f"[{crit_count} critical issue{'s' if crit_count != 1 else ''} found]",
        color="white", fontsize=16, y=0.995)

    fig.text(0.5, 0.01,
             "Synthetic analog mini-chip, planted ground-truth issue (matched BJT pair under RDL strip). "
             "Detection + annotation generated by the Qwen-powered pipeline "
             "(same pipeline, LLM backend swapped from Claude to a local Qwen server).",
             ha="center", color="#888888", fontsize=9, style="italic")

    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {out_path}")


def compute_issue_bounds(report):
    """Union bounds over all critical findings, so the zoom covers them all."""
    crit_items = []
    for g in report["groups"]:
        if g["sensitivity"] == "critical":
            crit_items.extend(g["items"])
    if not crit_items:
        return None
    minx = min(it["overlap_bounds"][0] for it in crit_items)
    miny = min(it["overlap_bounds"][1] for it in crit_items)
    maxx = max(it["overlap_bounds"][2] for it in crit_items)
    maxy = max(it["overlap_bounds"][3] for it in crit_items)
    return minx, miny, maxx, maxy


def _fallback_analysis(report):
    """Used when the LLM server isn't reachable. Same prose the Claude-
    powered demo originally had."""
    crit = [g for g in report["groups"] if g["sensitivity"] == "critical"]
    if not crit:
        return "No critical findings."
    device_type = crit[0]["instance_path"].split(" / ")[-1].split("@")[0]
    n_devices = len(crit)
    parent = crit[0]["instance_path"].split(" / ")[-2] if " / " in crit[0]["instance_path"] else ""
    total_area = sum(g["total_overlap_area"] for g in crit)
    return "\n".join([
        f"Issue: RDL routing crosses {n_devices} {device_type} devices in",
        f"the {parent} block. These instances are the matched",
        f"pair Q1 / Q2 that generate the bandgap PTAT voltage —",
        f"the output Vbg depends directly on their Vbe match.",
        "",
        f"Total overlap: {total_area:.0f} µm² RDL above BJT active area.",
        "",
        "Why it's a real problem (not geometric noise):",
        "",
        "  1. Thermal asymmetry. RDL current heats one device more",
        "     than the other; Vbe drifts −2 mV/°C.",
        "",
        "  2. Piezo-resistive stress. Thick RDL has CTE mismatch",
        "     with Si; edge vs. centre stress asymmetric.",
        "",
        "  3. Capacitive coupling. BGR base is high-Z; any",
        "     switching on the RDL couples straight onto Vbg.",
        "",
        "Recommended fix: re-route RDL off the BGR footprint, or",
        "add an M-top shield tied to quiet AGND with ≥5 µm margin.",
    ])


def build_ai_analysis(report, llm_url, llm_model):
    """Ask Qwen (or any OpenAI-compatible endpoint) for the narrative.
    Falls back to the hardcoded Claude-era prose if the call fails."""
    crit = [g for g in report["groups"] if g["sensitivity"] == "critical"]
    if not crit:
        return "No critical findings.", False

    device_type = crit[0]["instance_path"].split(" / ")[-1].split("@")[0]
    n_devices = len(crit)
    parent = crit[0]["instance_path"].split(" / ")[-2] if " / " in crit[0]["instance_path"] else ""
    total_area = sum(g["total_overlap_area"] for g in crit)

    prompt = (
        "Detector report:\n"
        f"  Block:  {parent}\n"
        f"  Devices under RDL strip: {n_devices} × {device_type}\n"
        f"  Total RDL-over-device overlap area: {total_area:.0f} µm²\n"
        f"  The devices form the matched Q1/Q2 pair of a bandgap reference "
        f"(PTAT voltage generation).\n"
        "\nWrite the review now in the required 4-5 paragraph format."
    )

    model = llm_model or _discover_model(llm_url)
    if not model:
        return _fallback_analysis(report), False

    print(f"  Calling LLM {llm_url}  model={model} ...", end="", flush=True)
    text = _call_llm(prompt, llm_url, model)
    if text:
        print(" ok")
        return text, True
    print(" fallback")
    return _fallback_analysis(report), False


def highlight_subckt(body):
    """Return the subckt body as a string with Q1/Q2 lines visually marked."""
    out = []
    for line in body:
        if re.search(r"\bXQ[12]\b", line):
            out.append(">> " + line + "  <-- matched pair")
        else:
            out.append("   " + line)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gds", default="demo_chip.gds")
    p.add_argument("--config", default="layer_config.json")
    p.add_argument("--findings", default="findings.json")
    p.add_argument("--netlist", default="demo_chip.cdl")
    p.add_argument("--output", default="demo_report.png")
    p.add_argument("--llm-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default=None,
                   help="default: auto-discover from /v1/models")
    args = p.parse_args()
    render(args.gds, args.config, args.findings, args.netlist, args.output,
           args.llm_url, args.llm_model)


if __name__ == "__main__":
    main()
