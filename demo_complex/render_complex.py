#!/usr/bin/env python3
"""
Render the complex_chip review report.

Two backends, same renderer:

  --backend claude     Read pre-authored Claude prose from
                       claude_responses.json. The Claude responses
                       cluster duplicate findings, flag the planted
                       false-positive explicitly, and stay terse.

  --backend qwen       Live OpenAI-compatible call to a local Qwen
                       server (sglang at localhost:8000 by default).
                       Treats every finding independently — there's
                       no cache to teach it which duplicates fold or
                       which finding is the planted false positive.

The visual layout is identical across backends so a side-by-side
comparison shows differences in *content*, not chrome.
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


# ── Layer styling — extends demo_synthetic with new layers ──────────────
STYLE = {
    (1, 0):   ("#2e7d32", 0.6, 1, "diff"),
    (2, 0):   ("#558b2f", 0.25, 0, "nwell"),
    (5, 0):   ("#c62828", 0.6, 2, "poly"),
    (6, 0):   ("#ff8a65", 0.55, 2, "poly_fill"),
    (10, 0):  ("#37474f", 0.8, 3, "contact"),
    (11, 0):  ("#1976d2", 0.45, 4, "met1"),
    (22, 0):  ("#8e24aa", 0.55, 5, "met2"),
    (33, 0):  ("#ef6c00", 0.30, 6, "met3"),
    (33, 1):  ("#aaaaaa", 0.0, 0, "outline"),
    (40, 0):  ("#ffd600", 0.50, 7, "met_thick"),
    (50, 0):  ("#ffeb3b", 0.40, 8, "RDL"),
    (60, 0):  ("#000000", 0.9,  9, "pad_open"),
    (82, 5):  ("#ad1457", 0.45, 6, "BJT"),
    (90, 0):  ("#00bcd4", 0.30, 6, "HI_Z"),
}


SEV_COLOR = {
    "critical":      "#ff1744",
    "high":          "#ff9100",
    "medium":        "#ffeb3b",
    "low":           "#9e9e9e",
    "false_positive":"#00bcd4",
}


# ── Geometry / rendering helpers ────────────────────────────────────────

def flatten_polygons(cell, origin=(0.0, 0.0), by_layer=None):
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


def get_bounds(cell):
    bb = cell.bounding_box()
    if bb is None:
        return 0, 0, 1000, 1000
    (mnx, mny), (mxx, mxy) = bb
    return mnx, mny, mxx, mxy


def render_layout(ax, top, draw_labels=True, label_fontsize=5):
    by_layer = flatten_polygons(top)
    for (layer, dt), polys in sorted(by_layer.items(),
                                     key=lambda kv: STYLE.get(kv[0], ("", 0, 0, ""))[2]):
        style = STYLE.get((layer, dt))
        if style is None:
            continue
        color, alpha, _, _ = style
        for pts in polys:
            if alpha == 0.0:
                mp = MplPolygon(pts, closed=True, fill=False,
                                edgecolor=color, linewidth=0.6, alpha=0.5)
            else:
                mp = MplPolygon(pts, closed=True, facecolor=color, alpha=alpha,
                                edgecolor=color, linewidth=0.4)
            ax.add_patch(mp)

    if draw_labels:
        for lbl in top.labels:
            x, y = float(lbl.origin[0]), float(lbl.origin[1])
            ax.text(x, y, lbl.text, fontsize=label_fontsize, color="white",
                    ha="center", va="center",
                    bbox=dict(facecolor="#000000", alpha=0.55, pad=0.6,
                              edgecolor="none"))

    minx, miny, maxx, maxy = get_bounds(top)
    pad = max(maxx - minx, maxy - miny) * 0.02
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("white")


def draw_finding_circles(ax, groups):
    for g in groups:
        sev = g["sensitivity"]
        color = SEV_COLOR.get(sev, "#ffffff")
        for it in g["items"]:
            mnx, mny, mxx, mxy = it["overlap_bounds"]
            cx = (mnx + mxx) / 2
            cy = (mny + mxy) / 2
            r = max(mxx - mnx, mxy - mny) / 2 + 18
            ax.add_patch(patches.Circle((cx, cy), radius=r, fill=False,
                                        edgecolor=color, linewidth=2.0,
                                        zorder=20))

    handles = []
    for sev, color in [("critical", SEV_COLOR["critical"]),
                       ("high",     SEV_COLOR["high"]),
                       ("medium",   SEV_COLOR["medium"])]:
        if any(g["sensitivity"] == sev for g in groups):
            handles.append(Line2D([0], [0], marker="o", color=color, linewidth=0,
                                  markerfacecolor="none", markeredgewidth=2,
                                  markersize=10, label=sev.upper()))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  facecolor="#222222", edgecolor="white", labelcolor="white")


def compute_zoom_bounds(group):
    items = group["items"]
    minx = min(it["overlap_bounds"][0] for it in items)
    miny = min(it["overlap_bounds"][1] for it in items)
    maxx = max(it["overlap_bounds"][2] for it in items)
    maxy = max(it["overlap_bounds"][3] for it in items)
    return minx, miny, maxx, maxy


# ── Backends ────────────────────────────────────────────────────────────

def group_key(g):
    return f"{g['rule']}__{g['block']}"


def claude_backend(report, cache_path):
    """Return (summary_text, [(group, text, headline, severity_label), ...])."""
    with open(cache_path) as f:
        cache = json.load(f)
    summary = cache.get("summary", "")
    findings_cache = cache.get("findings", {})

    cards = []
    for g in report["groups"]:
        key = group_key(g)
        entry = findings_cache.get(key)
        if entry is None:
            cards.append((g, "(no Claude response cached for this finding)",
                          f"{g['rule']} in {g['block']}", g["sensitivity"]))
            continue
        if "cluster_with" in entry:
            parent_key = entry["cluster_with"]
            parent = findings_cache.get(parent_key, {})
            note = entry.get("_cluster_note", "Same root cause as another finding above.")
            headline = f"[clustered] also fires in {g['block']}"
            body = f"{note}\n\nSee {parent.get('headline', parent_key)} above for the analysis."
            cards.append((g, body, headline, "clustered"))
        else:
            cards.append((g, entry.get("body", ""),
                          entry.get("headline", f"{g['rule']} in {g['block']}"),
                          entry.get("severity", g["sensitivity"])))
    return summary, cards


# ── Qwen / OpenAI-compatible backend ────────────────────────────────────

QWEN_SYSTEM = (
    "You are an analog/mixed-signal IC layout reviewer. "
    "Given a single detector finding, write a short review in 2-4 short "
    "paragraphs separated by blank lines. Plain prose only — no headers, "
    "no markdown, no bullet lists, no preamble. Start with the analysis "
    "directly. Be specific about the physical mechanism and the fix."
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
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
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


def qwen_backend(report, llm_url, llm_model):
    model = llm_model or _discover_model(llm_url)
    if not model:
        print("Qwen server not reachable — falling back to per-finding template prose.",
              file=sys.stderr)
        cards = []
        for g in report["groups"]:
            cards.append((g,
                          f"{g['why']}\n\n(Qwen unreachable — template fallback used.)",
                          f"{g['rule']} in {g['block']}", g["sensitivity"]))
        return ("Qwen server not reachable.", cards)

    cards = []
    for g in report["groups"]:
        prompt = (
            f"Detector finding:\n"
            f"  Rule: {g['rule']}\n"
            f"  Block: {g['block']}\n"
            f"  Aggressor layer: {g['aggressor_label']}\n"
            f"  Victim layer: {g['victim_label']}\n"
            f"  Severity (heuristic): {g['sensitivity']}\n"
            f"  Number of overlap polygons: {g['count']}\n"
            f"  Total overlap area: {g['total_overlap_area']} square micrometers\n"
            f"  Instance path: {g['instance_path']}\n"
            f"  Detector justification: {g['why']}\n\n"
            "Write the review now. Keep it to 2-4 short paragraphs."
        )
        print(f"  Qwen: {g['rule']} / {g['block']} ...", end="", flush=True)
        text = _call_llm(prompt, llm_url, model)
        if text:
            print(" ok")
        else:
            print(" failed (using detector justification)")
            text = g["why"]
        cards.append((g, text, f"{g['rule']} in {g['block']}", g["sensitivity"]))

    summary = (
        f"Qwen review of {len(report['groups'])} findings on {report['gds_file']}. "
        "Each finding analyzed independently."
    )
    return summary, cards


# ── Netlist evidence ────────────────────────────────────────────────────

def extract_subckt_body(netlist_path, subckt_name):
    if not os.path.exists(netlist_path):
        return [f"(netlist {netlist_path} not found)"]
    body = []
    inside = False
    with open(netlist_path) as f:
        for line in f:
            s = line.rstrip()
            sl = s.strip().lower()
            if not inside and sl.startswith(".subckt"):
                parts = s.strip().split()
                if len(parts) >= 2 and parts[1].lower() == subckt_name.lower():
                    inside = True
                    body.append(s.strip())
                    continue
            if inside:
                body.append(s.strip())
                if sl.startswith(".ends"):
                    break
    return body if body else [f"(no .SUBCKT {subckt_name} in netlist)"]


# ── Main render ─────────────────────────────────────────────────────────

def render(report, cards, summary, top_cell, netlist_path, out_path,
           backend_label):
    n_cards = len(cards)
    fig_height = max(14, 5 + 2.4 * n_cards)
    fig = plt.figure(figsize=(22, fig_height), facecolor="#0e0e0e")
    gs = fig.add_gridspec(
        n_cards + 1, 2,
        width_ratios=[1.5, 1.6],
        height_ratios=[2.2] + [1.0] * n_cards,
        wspace=0.06, hspace=0.35,
    )

    # Full layout view (top-left, spans the layout row only)
    ax_full = fig.add_subplot(gs[0, 0])
    render_layout(ax_full, top_cell, draw_labels=True, label_fontsize=6)
    draw_finding_circles(ax_full, report["groups"])
    ax_full.set_title(f"GDS layout — {top_cell.name}",
                      color="white", fontsize=12)

    # Summary panel (top-right)
    ax_sum = fig.add_subplot(gs[0, 1])
    ax_sum.axis("off")
    ax_sum.set_facecolor("#0e0e0e")
    ax_sum.set_xlim(0, 1); ax_sum.set_ylim(0, 1)
    ax_sum.text(0.0, 1.0, f"REVIEW SUMMARY — {backend_label}",
                fontsize=14, weight="bold", color="#00e5ff",
                va="top", transform=ax_sum.transAxes)
    sev_counts = {}
    for g in report["groups"]:
        sev_counts[g["sensitivity"]] = sev_counts.get(g["sensitivity"], 0) + 1
    badge_line = "   ".join(
        f"{n} {sev.upper()}" for sev, n in sorted(sev_counts.items())
    )
    ax_sum.text(0.0, 0.91, badge_line,
                fontsize=10, color="#bbbbbb", weight="bold",
                va="top", transform=ax_sum.transAxes)
    ax_sum.text(0.0, 0.84, summary,
                fontsize=10, color="white",
                va="top", transform=ax_sum.transAxes,
                wrap=True)

    # One row per finding card
    for row, (g, body, headline, sev_label) in enumerate(cards, start=1):
        # Left: zoomed view of this finding
        ax_z = fig.add_subplot(gs[row, 0])
        render_layout(ax_z, top_cell, draw_labels=False)
        draw_finding_circles(ax_z, [g])
        mnx, mny, mxx, mxy = compute_zoom_bounds(g)
        pad = max(mxx - mnx, mxy - mny) * 0.7 + 25
        ax_z.set_xlim(mnx - pad, mxx + pad)
        ax_z.set_ylim(mny - pad, mxy + pad)
        ax_z.set_title(
            f"{g['block']} — {g['aggressor_label']} over {g['victim_label']}  "
            f"({g['count']} overlaps, {g['total_overlap_area']:.0f} um²)",
            color=SEV_COLOR.get(g["sensitivity"], "white"),
            fontsize=9)

        # Right: prose card
        ax_t = fig.add_subplot(gs[row, 1])
        ax_t.axis("off")
        ax_t.set_facecolor("#0e0e0e")
        ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

        sev_color = SEV_COLOR.get(sev_label, "#ffffff") \
            if sev_label != "clustered" else "#9e9e9e"
        # Severity badge
        ax_t.text(0.0, 0.97, f"  {sev_label.upper()}  ",
                  fontsize=9, weight="bold", color="#0e0e0e",
                  va="top", ha="left",
                  bbox=dict(facecolor=sev_color, edgecolor=sev_color, pad=2),
                  transform=ax_t.transAxes)
        # Headline
        ax_t.text(0.18, 0.965, headline,
                  fontsize=11, weight="bold", color="white",
                  va="top", ha="left", transform=ax_t.transAxes)
        # Body
        ax_t.text(0.0, 0.85, body,
                  fontsize=8.5, color="white", family="DejaVu Sans",
                  va="top", ha="left", transform=ax_t.transAxes,
                  wrap=True)

    fig.suptitle(
        f"AI layout review — {report['gds_file']}   "
        f"[backend: {backend_label}]",
        color="white", fontsize=18, y=0.995)

    fig.text(0.5, 0.005,
             "Synthetic complex chip with 4 planted issues + 1 expected pad route. "
             "Same detector + same renderer; backend swapped to compare reasoning.",
             ha="center", color="#888888", fontsize=9, style="italic")

    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gds", default="complex_chip.gds")
    p.add_argument("--findings", default="findings.json")
    p.add_argument("--netlist", default="complex_chip.cdl")
    p.add_argument("--backend", choices=["claude", "qwen"], required=True)
    p.add_argument("--cache", default="claude_responses.json",
                   help="Path to Claude response cache (used when backend=claude).")
    p.add_argument("--llm-url",
                   default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--output", default=None,
                   help="Default: complex_report_<backend>.png")
    args = p.parse_args()

    out_path = args.output or f"complex_report_{args.backend}.png"

    print(f"Loading {args.gds} ...")
    lib = gdstk.read_gds(args.gds)
    top = lib.top_level()[0]

    with open(args.findings) as f:
        report = json.load(f)

    if args.backend == "claude":
        summary, cards = claude_backend(report, args.cache)
        backend_label = "Claude (cached)"
    else:
        summary, cards = qwen_backend(report, args.llm_url, args.llm_model)
        backend_label = "Qwen (live)"

    render(report, cards, summary, top, args.netlist, out_path, backend_label)


if __name__ == "__main__":
    main()
