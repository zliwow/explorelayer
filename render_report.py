#!/usr/bin/env python3
"""
render_report.py — visual + LLM report for findings_v2.json

Mirrors the gdsattemptclaude/render_report.py demo on the real chip.
Builds a single PNG with:
  - a chip-layout panel (chip outline + red boxes on top findings)
  - a zoomed inset on the worst finding (real polygons from the GDS)
  - an analysis panel: one LLM-written blurb per finding

LLM backend is OpenAI-compatible chat-completions. Default URL is the
work machine's Qwen server at http://localhost:8000/v1/chat/completions.
If the request fails (no server / wrong URL / non-200), each finding
falls back to a deterministic template so the report still renders.

Usage:
  python render_report.py findings_v2.json examples/mpq8897r4_1.gds
  python render_report.py findings_v2.json examples/mpq8897r4_1.gds \
      --output report_v2.png --top 5 \
      --llm-url http://localhost:8000/v1/chat/completions \
      --llm-model qwen
"""

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request

import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


TIER_COLOR = {
    "high":   "#d62728",  # red
    "medium": "#ff7f0e",  # orange
    "low":    "#1f77b4",  # blue
    "info":   "#7f7f7f",  # grey
}
TIER_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}


# ── Loading ────────────────────────────────────────────────────────────

def load_findings(path):
    with open(path) as f:
        return json.load(f)


def pick_top_findings(report, n):
    """Take top-N groups, skipping 'info' if any actionable groups exist."""
    groups = report.get("groups", [])
    actionable = [g for g in groups if g["severity_tier"] != "info"]
    pool = actionable if actionable else groups
    pool = sorted(pool,
                  key=lambda g: (-TIER_RANK.get(g["severity_tier"], 0),
                                 -g["total_overlap_area"]))
    return pool[:n]


def attach_geometry(top_groups, raw_findings):
    """Each group references finding indices — attach the actual bboxes."""
    for g in top_groups:
        bboxes = []
        for i in g.get("sample_finding_indices", []):
            if 0 <= i < len(raw_findings):
                f = raw_findings[i]
                if "overlap_bounds" in f:
                    bboxes.append(("overlap", f["overlap_bounds"]))
                elif "region_bounds" in f:
                    bboxes.append(("region", f["region_bounds"]))
        g["_bboxes"] = bboxes
    return top_groups


# ── LLM call (OpenAI-compatible chat completions) ──────────────────────

LLM_SYSTEM = (
    "You are an analog/mixed-signal layout reviewer. Given one detector "
    "finding (a metal/poly polygon overlap detected under a pad or RDL), "
    "write a 3-4 sentence explanation aimed at the layout (LO) and design "
    "(DE) engineers. Cover: (1) what the affected cell likely is, "
    "(2) why this overlap could matter (mechanical stress, capacitive "
    "coupling, ESD/discharge path, matching, current density), and "
    "(3) what to check or do next. Be specific. No filler. Plain text only."
)


def discover_model(llm_url, timeout=5):
    """GET /v1/models (derived from the chat URL) and return the first model id.
    Returns None if the server isn't reachable or response isn't usable."""
    if "/chat/completions" in llm_url:
        models_url = llm_url.replace("/chat/completions", "/models")
    elif llm_url.rstrip("/").endswith("/v1"):
        models_url = llm_url.rstrip("/") + "/models"
    else:
        models_url = llm_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        if data and isinstance(data, list):
            return data[0].get("id")
    except Exception:
        return None
    return None


def call_llm(prompt, url, model, timeout=30):
    """POST to /v1/chat/completions and return the assistant text. Returns
    None on any failure so callers fall back to template text."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 350,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload["choices"][0]["message"]
        # Qwen3 thinking models put the answer in reasoning_content and leave
        # content = null. Prefer real content; fall back to reasoning_content.
        text = msg.get("content") or msg.get("reasoning_content") or ""
        text = text.strip()
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError,
            KeyError, TypeError, AttributeError,
            json.JSONDecodeError, TimeoutError) as exc:
        print(f"  LLM call failed: {exc}", file=sys.stderr)
        return None


def build_prompt(group):
    reasons = "; ".join(group.get("severity_reasons", [])) or "no specific keyword match"
    bb = group["_bboxes"][0][1] if group.get("_bboxes") else None
    bb_str = f"approx ({bb[0]:.1f},{bb[1]:.1f})–({bb[2]:.1f},{bb[3]:.1f}) µm" if bb else "n/a"
    return (
        f"Finding:\n"
        f"  Owner cell path: {group['owner_path']}\n"
        f"  Affected layer:  {group['check_layer']}\n"
        f"  Overlap count:   {group['count']}\n"
        f"  Total overlap area: {group['total_overlap_area']} µm²\n"
        f"  Detector severity tier: {group['severity_tier']}\n"
        f"  Heuristic reasons: {reasons}\n"
        f"  Approx location: {bb_str}\n"
        f"\n"
        f"Write the 3-4 sentence review now."
    )


def template_blurb(group):
    """Deterministic fallback when the LLM is unavailable."""
    leaf = (group["owner_path"] or "").rsplit("/", 1)[-1] or "(unknown)"
    tier = group["severity_tier"]
    cnt = group["count"]
    area = group["total_overlap_area"]
    layer = group["check_layer"]
    reasons = group.get("severity_reasons") or ["no name-pattern match"]
    return (
        f"{cnt} overlap polygons on layer {layer} attribute to cell "
        f"'{leaf}' (severity={tier}, total area={area:.1f} µm²). "
        f"Heuristic flags: {reasons[0]}. "
        f"Recommend the layout owner confirm whether this metal coverage "
        f"is intentional (power routing / shielding) or accidental "
        f"(top-level interconnect crossing a sensitive device)."
    )


# ── Rendering ──────────────────────────────────────────────────────────

def chip_bbox(top_cell):
    bb = top_cell.bounding_box()
    if bb is None:
        return (0.0, 0.0, 1.0, 1.0)
    (x0, y0), (x1, y1) = bb
    return (float(x0), float(y0), float(x1), float(y1))


def polygons_in_bbox(top_cell, layers, bbox, max_polys=4000):
    """Yield (layer, datatype, points) for polygons that intersect bbox.
    Caps total polygons to keep render time bounded."""
    x0, y0, x1, y1 = bbox
    out = []
    for ld in layers:
        layer, dt = ld
        polys = top_cell.get_polygons(layer=layer, datatype=dt)
        for gp in polys:
            pts = gp.points if isinstance(gp, gdstk.Polygon) else gp
            if len(pts) < 3:
                continue
            # quick bbox test
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
                continue
            out.append((layer, dt, pts))
            if len(out) >= max_polys:
                return out
    return out


LAYER_COLOR_CYCLE = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
                     "#e377c2", "#bcbd22", "#17becf", "#ff9896"]


def render_chip_panel(ax, chip_bb, top_groups):
    x0, y0, x1, y1 = chip_bb
    pad = 0.02 * max(x1 - x0, y1 - y0)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, edgecolor="#333", linewidth=1.0))
    for i, g in enumerate(top_groups, 1):
        color = TIER_COLOR.get(g["severity_tier"], "#666")
        for kind, bb in g.get("_bboxes", []):
            bx0, by0, bx1, by1 = bb
            w, h = bx1 - bx0, by1 - by0
            # Boxes can be tiny relative to die; pad them so they're visible
            min_size = 0.005 * max(x1 - x0, y1 - y0)
            if w < min_size:
                bx0 -= (min_size - w) / 2
                w = min_size
            if h < min_size:
                by0 -= (min_size - h) / 2
                h = min_size
            ax.add_patch(Rectangle((bx0, by0), w, h,
                                   fill=False, edgecolor=color, linewidth=1.4))
        # numeric label at first bbox
        if g.get("_bboxes"):
            bb0 = g["_bboxes"][0][1]
            ax.text(bb0[0], bb0[3], f"#{i}",
                    color=color, fontsize=9, fontweight="bold",
                    ha="left", va="bottom")
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_title("Chip layout — top findings highlighted",
                 fontsize=11, loc="left")
    ax.set_xticks([]); ax.set_yticks([])


def render_inset_panel(ax, top_cell, top_groups, layer_lookup):
    """Zoom into the #1 finding's bbox; draw real polygons from involved
    layers for context."""
    if not top_groups or not top_groups[0].get("_bboxes"):
        ax.text(0.5, 0.5, "No bbox data for inset", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        return

    target = top_groups[0]
    bb = target["_bboxes"][0][1]
    bx0, by0, bx1, by1 = bb
    w, h = bx1 - bx0, by1 - by0
    margin = max(w, h) * 1.5 + 5.0
    rx0, ry0 = bx0 - margin, by0 - margin
    rx1, ry1 = bx1 + margin, by1 + margin

    # Layers to render in inset: the check layer plus pad/RDL layers
    layers = []
    layer_name = target["check_layer"]
    if layer_name in layer_lookup:
        layers.append(layer_lookup[layer_name])
    layers.extend(layer_lookup.get("__pad_rdl__", []))
    polys = polygons_in_bbox(top_cell, layers, (rx0, ry0, rx1, ry1),
                             max_polys=4000)
    layer_to_color = {}
    for layer, dt, pts in polys:
        key = (layer, dt)
        if key not in layer_to_color:
            layer_to_color[key] = LAYER_COLOR_CYCLE[len(layer_to_color) % len(LAYER_COLOR_CYCLE)]
        color = layer_to_color[key]
        ax.fill([p[0] for p in pts], [p[1] for p in pts],
                color=color, alpha=0.4, linewidth=0)

    # Highlight the finding box
    ax.add_patch(Rectangle((bx0, by0), w, h,
                           fill=False, edgecolor=TIER_COLOR.get(target["severity_tier"], "#d62728"),
                           linewidth=2.0))
    ax.text(bx0, by1, "#1", color=TIER_COLOR.get(target["severity_tier"], "#d62728"),
            fontweight="bold", fontsize=10, ha="left", va="bottom")
    ax.set_xlim(rx0, rx1)
    ax.set_ylim(ry0, ry1)
    ax.set_aspect("equal")
    leaf = (target["owner_path"] or "").rsplit("/", 1)[-1] or "(unknown)"
    ax.set_title(f"Inset: {leaf} / {target['check_layer']}",
                 fontsize=11, loc="left")
    ax.set_xticks([]); ax.set_yticks([])


def render_analysis_panel(ax, top_groups, blurbs, llm_used):
    ax.axis("off")
    title = "AI analysis" + (" (Qwen)" if llm_used else " (template fallback — LLM unavailable)")
    ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold",
            ha="left", va="top", transform=ax.transAxes)
    y = 0.93
    for i, (g, blurb) in enumerate(zip(top_groups, blurbs), 1):
        leaf = (g["owner_path"] or "").rsplit("/", 1)[-1] or "(unknown)"
        head = (f"#{i}  [{g['severity_tier']}]  {leaf}  /  {g['check_layer']}  "
                f"·  count={g['count']}  area={g['total_overlap_area']:.1f}")
        ax.text(0.0, y, head, fontsize=9.5, fontweight="bold",
                ha="left", va="top", transform=ax.transAxes,
                color=TIER_COLOR.get(g["severity_tier"], "#000"))
        y -= 0.035
        wrapped = textwrap.fill(blurb, width=145)
        ax.text(0.01, y, wrapped, fontsize=8.5,
                ha="left", va="top", transform=ax.transAxes,
                family="monospace")
        y -= 0.035 * (wrapped.count("\n") + 1) + 0.02
        if y < 0.05:
            ax.text(0.0, y, "  … (more findings truncated; see findings_v2.json)",
                    fontsize=8, ha="left", va="top", transform=ax.transAxes)
            break


# ── Main ───────────────────────────────────────────────────────────────

def autofind(path_candidates, kind):
    """Return the first existing path from a list, else None."""
    for p in path_candidates:
        if p and os.path.exists(p):
            return p
    print(f"Could not auto-find {kind}. Pass it as a positional argument.")
    return None


def autofind_gds():
    import glob
    for pat in ("examples/*.gds", "*.gds", "**/*.gds"):
        matches = sorted(glob.glob(pat, recursive=True))
        if matches:
            return matches[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("findings_json", nargs="?", default=None,
                   help="default: findings_v2.json")
    p.add_argument("gds_file", nargs="?", default=None,
                   help="default: first .gds under examples/ or cwd")
    p.add_argument("--output", default="report_v2.png")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--llm-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default=None,
                   help="default: auto-discover from /v1/models")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM call and always use the template blurb.")
    args = p.parse_args()

    findings_path = args.findings_json or autofind(
        ["findings_v2.json", "findings.json"], "findings JSON")
    gds_path = args.gds_file or autofind_gds()
    if not findings_path or not gds_path:
        sys.exit(1)
    if not args.findings_json:
        print(f"[auto] findings: {findings_path}")
    if not args.gds_file:
        print(f"[auto] gds:      {gds_path}")

    # Auto-discover model id if not provided
    llm_model = args.llm_model
    if not args.no_llm and not llm_model:
        print(f"Discovering model id from {args.llm_url.replace('/chat/completions', '/models')} ...")
        llm_model = discover_model(args.llm_url)
        if llm_model:
            print(f"[auto] llm-model: {llm_model}")
        else:
            print("  server not reachable — using fallback template for all findings")

    print(f"Loading {findings_path} ...")
    report = load_findings(findings_path)
    raw = report.get("findings", [])
    top_groups = pick_top_findings(report, args.top)
    if not top_groups:
        print("No findings to render.")
        sys.exit(1)
    attach_geometry(top_groups, raw)

    print(f"Loading {gds_path} ...")
    t0 = time.time()
    lib = gdstk.read_gds(gds_path)
    top = lib.top_level()
    if not top:
        print("No top cell.")
        sys.exit(1)
    top_cell = top[0]
    print(f"  loaded in {time.time() - t0:.1f}s, top cell: {top_cell.name}")
    chip_bb = chip_bbox(top_cell)

    # Map check_layer name → (layer, datatype) for inset rendering
    cfg = report.get("config", {})
    layer_lookup = {c["name"]: tuple(c["layer"]) for c in cfg.get("check_layers", [])}
    layer_lookup["__pad_rdl__"] = ([tuple(l) for l in cfg.get("pad_layers", [])]
                                   + [tuple(l) for l in cfg.get("rdl_layers", [])])

    # Per-finding blurbs
    print(f"Building analyses for {len(top_groups)} findings ...")
    blurbs = []
    llm_used = False
    for i, g in enumerate(top_groups, 1):
        if args.no_llm:
            blurbs.append(template_blurb(g))
            continue
        prompt = build_prompt(g)
        print(f"  [{i}/{len(top_groups)}] calling LLM ...", end="", flush=True)
        text = call_llm(prompt, args.llm_url, llm_model) if llm_model else None
        if text:
            print(" ok")
            blurbs.append(text)
            llm_used = True
        else:
            print(" fallback")
            blurbs.append(template_blurb(g))

    print("Rendering figure ...")
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1],
                  hspace=0.18, wspace=0.12)
    ax_chip   = fig.add_subplot(gs[0, 0])
    ax_inset  = fig.add_subplot(gs[0, 1])
    ax_text   = fig.add_subplot(gs[1, :])

    render_chip_panel(ax_chip, chip_bb, top_groups)
    render_inset_panel(ax_inset, top_cell, top_groups, layer_lookup)
    render_analysis_panel(ax_text, top_groups, blurbs, llm_used)

    fig.suptitle(
        f"GDS layout review — {os.path.basename(gds_path)} "
        f"·  {len(raw)} raw findings → {len(top_groups)} top groups",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
