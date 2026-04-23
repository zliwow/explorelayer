#!/usr/bin/env python3
"""
stage2_reasoner.py — stage 2 of the v5 pipeline.

Reads extraction.json, runs focused LLM calls against a local Qwen
endpoint at http://localhost:8000/v1, and writes semantics.json:

  2A  layers        per-layer semantic role (RDL, top_metal, pad, diff,
                    poly, BJT_marker, passivation, …) with confidence +
                    evidence. Input: chip + layers + layer_pair_overlaps
                    + per-layer unique label texts.

  2B  cells         (scaffolded, not yet wired) per-top-cell role
                    (bandgap, LDO, comparator, oscillator, …) with
                    confidence + evidence.

  2C  candidates    (scaffolded, not yet wired) aggressor→victim triples
                    worth deterministic checking in stage 3.

Each section is a separate LLM call so the full context stays focused
and well under Qwen's 131k budget. Sections are cached to semantics.json
incrementally — rerunning re-does only what's requested.

Usage:
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json --sections 2a
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json --output demo_v5/semantics.json
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json --llm-url http://localhost:8000
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request


# ── LLM backend (Qwen local, OpenAI-compatible) ────────────────────────

def discover_model(base_url, timeout=5):
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        return data[0].get("id") if data else None
    except Exception:
        return None


def call_llm(system_prompt, user_prompt, base_url, model,
             temperature=0.1, max_tokens=4000, timeout=600):
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = payload["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_block(text):
    """
    Extract the first JSON object from an LLM response. Handles fenced
    ```json blocks and bare objects.
    """
    text = strip_thinking(text)
    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # bare { ... }
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    return text


def call_llm_json(system_prompt, user_prompt, base_url, model, **kw):
    raw = call_llm(system_prompt, user_prompt, base_url, model, **kw)
    body = extract_json_block(raw)
    try:
        return json.loads(body), raw
    except json.JSONDecodeError as e:
        print(f"  !! JSON decode failed: {e}", file=sys.stderr)
        print(f"  raw response (first 1000 chars):\n{raw[:1000]}", file=sys.stderr)
        return None, raw


# ── Section 2A: layer semantics ─────────────────────────────────────────

SECTION_2A_SYSTEM = """You are an IC layout semantics classifier. You read structured feature data from a GDS file and identify the semantic role of each layer.

Roles to consider: RDL, top_metal, intermediate_metal, pad_opening, passivation, bump, diffusion, poly, implant, nwell, pwell, via, contact, BJT_marker, MOS_marker, resistor_marker, capacitor_marker, fill, seal_ring, marker_text, boundary, unclassified.

Rules:
- Reason from the evidence provided: geometry fingerprint, pairwise layer overlap, unique label texts per layer.
- RDL typically sits on top of (covers) the pad/bump layer, has few large chunky polygons, high die coverage, and near the top of the metal stack.
- Pad/bump layers have many similar-sized rectangles near the die edge, bounded under passivation openings.
- BJT markers typically have few polygons and sit inside cells named BJT/QNPN/QPNP.
- If unsure, set role='unclassified' and say so — do NOT guess.
- Output MUST be a single JSON object, no prose before or after, no markdown fences.
"""

SECTION_2A_USER_TEMPLATE = """Classify every layer in this GDS.

# CHIP
{chip}

# LAYERS (inventory with geometry fingerprint)
{layers}

# LAYER-PAIR OVERLAPS (top pairs by intersection area)
{overlaps}

# LABEL UNIQUE TEXTS PER LAYER (first 100 unique strings, truncated)
{labels_summary}

Respond with a single JSON object in this exact shape:

{{
  "layers": [
    {{
      "id": "LAYER/DATATYPE",
      "role": "RDL | top_metal | pad_opening | passivation | diffusion | poly | BJT_marker | ... | unclassified",
      "confidence": "high | medium | low",
      "evidence": "one sentence citing specific numbers from the inputs above",
      "alternatives": ["other roles you considered with brief reason, may be empty"]
    }}
  ],
  "summary": {{
    "picked_rdl": "LAYER/DATATYPE or null",
    "picked_top_metal": "LAYER/DATATYPE or null",
    "picked_pad_opening": "LAYER/DATATYPE or null",
    "picked_bjt_marker": "LAYER/DATATYPE or null",
    "notes": "any caveats or ambiguities in one sentence"
  }}
}}
"""


def build_2a_input(extraction):
    """
    Slim the extraction down to the signals section 2A needs.
    """
    # Chip
    chip = extraction["chip"]

    # Layers — keep the essentials, drop full stats dict
    layers = []
    for li in extraction["layers"]:
        s = li.get("stats") or {}
        layers.append({
            "id": li["id"],
            "weak_category": li["weak_category"],
            "polys": s.get("total_count", 0),
            "mean_area": round(s.get("mean_area", 0), 1),
            "area_pct_of_die": round(s.get("area_pct_of_die", 0), 2),
            "pct_rect": round(s.get("pct_rect", 0), 0),
            "pct_near_edge": round(s.get("pct_near_edge", 0), 0),
            "median_ar": round(s.get("median_ar", 0), 2),
        })

    # Top overlap pairs
    overlaps = extraction.get("layer_pair_overlaps", [])[:25]

    # Unique label texts per layer — cap at 100 per layer
    labels_summary = {}
    for layer_id, info in extraction.get("labels_by_layer", {}).items():
        uniq = info.get("unique_texts", [])
        labels_summary[layer_id] = {
            "n_labels": info.get("n_labels", 0),
            "n_unique": info.get("n_unique", 0),
            "sample_unique": uniq[:100],
        }

    return chip, layers, overlaps, labels_summary


def run_section_2a(extraction, base_url, model):
    chip, layers, overlaps, labels_summary = build_2a_input(extraction)
    user_prompt = SECTION_2A_USER_TEMPLATE.format(
        chip=json.dumps(chip, indent=2),
        layers=json.dumps(layers, indent=2),
        overlaps=json.dumps(overlaps, indent=2),
        labels_summary=json.dumps(labels_summary, indent=2),
    )

    n_tokens_approx = len(user_prompt) // 4
    print(f"  [2A] prompt: {len(user_prompt):,} chars "
          f"(~{n_tokens_approx:,} tokens)")
    t0 = time.time()
    result, raw = call_llm_json(SECTION_2A_SYSTEM, user_prompt, base_url, model,
                                max_tokens=8000)
    dt = time.time() - t0
    print(f"  [2A] LLM returned in {dt:.1f}s")
    if result is None:
        return {"error": "LLM did not return valid JSON", "raw": raw[:2000]}
    return result


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("extraction_json")
    ap.add_argument("--output", default=None,
                    help="Default: semantics.json next to extraction.json")
    ap.add_argument("--sections", default="2a",
                    help="Comma-separated list (2a,2b,2c). Default: 2a only.")
    ap.add_argument("--llm-url", default="http://localhost:8000")
    ap.add_argument("--llm-model", default=None,
                    help="Default: auto-discover from /v1/models.")
    args = ap.parse_args()

    if not os.path.exists(args.extraction_json):
        sys.exit(f"Not found: {args.extraction_json}")

    extraction = json.load(open(args.extraction_json))
    print(f"Loaded extraction: {extraction.get('gds_file')} "
          f"(schema v{extraction.get('schema_version')})")

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.extraction_json)), "semantics.json")
    semantics = {}
    if os.path.exists(out_path):
        semantics = json.load(open(out_path))
        print(f"Loaded existing semantics: {out_path} "
              f"(sections so far: {list(semantics.keys())})")

    model = args.llm_model or discover_model(args.llm_url)
    if not model:
        sys.exit(f"No model discovered at {args.llm_url}/v1/models — "
                 f"pass --llm-model explicitly.")
    print(f"Using LLM: {model} @ {args.llm_url}")

    sections = [s.strip().lower() for s in args.sections.split(",")]

    if "2a" in sections:
        print("\n=== Section 2A: layer semantics ===")
        semantics["2a_layers"] = run_section_2a(extraction, args.llm_url, model)
        with open(out_path, "w") as f:
            json.dump(semantics, f, indent=2)
        print(f"  -> wrote {out_path}")

    if "2b" in sections:
        print("\n=== Section 2B: cell roles (not implemented yet) ===")
        print("  (skipped — ship 2A, validate, then wire 2B)")

    if "2c" in sections:
        print("\n=== Section 2C: issue candidates (not implemented yet) ===")
        print("  (skipped — depends on 2A + 2B)")

    print(f"\nDone. Semantics so far: {list(semantics.keys())}")


if __name__ == "__main__":
    main()
