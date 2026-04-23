#!/usr/bin/env python3
"""
stage2_reasoner.py — stage 2 of the v5 pipeline.

Reads extraction.json, runs focused LLM calls against a local Qwen
endpoint at http://localhost:8000/v1, and writes semantics.json:

  2A  layers        per-layer semantic role (RDL, top_metal, pad, diff,
                    poly, BJT_marker, passivation, …) with confidence +
                    evidence. Input: chip + layers + layer_pair_overlaps
                    + per-layer unique label texts.

  2B  cells         per-top-cell functional role (bandgap, LDO, comparator,
                    oscillator, buck, digital, memory, I/O, ESD, …) with
                    confidence + evidence. Input: top cells (children,
                    parents, deep polys) + 2A output for context.

  2C  candidates    aggressor→victim triples worth deterministic checking
                    in stage 3 (e.g., RDL over BJT-marker inside bandgap).
                    Input: 2A + 2B + layer-pair overlaps + known mechanism
                    patterns.

Each section is a separate LLM call so the full context stays focused
and well under Qwen's 131k budget. Sections are cached to semantics.json
incrementally — rerunning re-does only what's requested. Default runs
the full 2A → 2B → 2C chain in one invocation.

Usage:
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json --sections 2a
    python demo_v5/stage2_reasoner.py demo_v5/extraction.json --sections 2a,2b,2c
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
             temperature=0.1, max_tokens=65000, timeout=3600,
             force_json=True):
    """
    Call Qwen3 with thinking ENABLED. Budget is generous because we own the
    endpoint and only care about the 131k context window, not runtime.
    Qwen3 returns chain-of-thought in `reasoning_content` and the final JSON
    in `content`; we prefer `content` so callers get the answer, not the CoT.
    """
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
    if force_json:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Some servers reject response_format — retry without it
        if force_json and e.code == 400:
            body.pop("response_format", None)
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        else:
            raise
    msg = payload["choices"][0]["message"]
    finish = payload["choices"][0].get("finish_reason")
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if finish == "length":
        print(f"  !! finish_reason=length — model hit max_tokens={max_tokens:,}. "
              f"content={len(content)} chars, reasoning={len(reasoning)} chars.",
              file=sys.stderr)
    # Prefer the final answer; fall back to reasoning only if content is empty
    # (which means the model ran out of tokens mid-think).
    return content or reasoning


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
    # Empty content sometimes happens when response_format=json_object conflicts
    # with Qwen3 thinking mode for a given prompt — retry once without the
    # grammar constraint before giving up.
    if not raw.strip() and kw.get("force_json", True):
        print("  !! empty response with force_json=True — retrying without "
              "response_format", file=sys.stderr)
        kw_retry = dict(kw)
        kw_retry["force_json"] = False
        raw = call_llm(system_prompt, user_prompt, base_url, model, **kw_retry)
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
    result, raw = call_llm_json(SECTION_2A_SYSTEM, user_prompt, base_url, model)
    dt = time.time() - t0
    print(f"  [2A] LLM returned in {dt:.1f}s")
    if result is None:
        return {"error": "LLM did not return valid JSON", "raw": raw[:2000]}
    return result


# ── Section 2B: cell functional roles ───────────────────────────────────

SECTION_2B_SYSTEM = """You are an IC block-level functional classifier. You read cell hierarchy and layer context and identify the functional role of each cell you are asked about.

Roles to consider: bandgap_reference, LDO, comparator, oscillator, buck_converter, boost_converter, charge_pump, digital, memory, io_pad, esd, bias_generator, reference, analog_mux, matched_pair_cell, level_shifter, opamp, power_switch, chip_top, unknown.

Rules:
- You MUST classify EVERY cell provided in the input list — do not skip, do not collapse, do not output only the chip's top cell. If the list has 40 entries you return 40 classifications.
- The chip's root/wrapper cell (is_top=true) is almost always 'chip_top', not 'unknown'. Non-top cells should be classified by their actual function.
- Reason from EVIDENCE: cell name tokens (BG/BANDGAP/VBG → bandgap_reference; LDO/VREG → LDO; OSC/VCO → oscillator; BUCK/SW → buck_converter; CMP/COMP → comparator; DIG/LOGIC → digital; PAD/IO → io_pad; ESD → esd), children names, n_instances (≥2 with same child → matched_pair_cell), deep polygon count (big analog blocks are 10k-100k polys; digital is often 100k+; small cells <1k polys are leaf devices).
- Tokens are strong evidence but not definitive — weigh them against children and n_instances.
- Section 2A output gives you which layers are RDL/BJT_marker/diff/poly — use that to interpret children.
- If unsure for a specific cell, set role='unknown' for that one cell — do NOT apply 'unknown' to the whole chip.
- Output MUST be a single JSON object, no prose before or after, no markdown fences.
"""

SECTION_2B_USER_TEMPLATE = """Classify each of the {n_cells} cells provided below. Return one classification per cell, in the same order.

# CHIP
{chip}

# SECTION 2A LAYER SEMANTICS (for context)
{semantics_2a}

# CELLS TO CLASSIFY (ranked by deep polygon count) — classify ALL {n_cells}
{cells}

Respond with a single JSON object in this exact shape:

{{
  "cells": [
    {{
      "name": "CELL_NAME",
      "role": "bandgap_reference | LDO | comparator | oscillator | buck_converter | charge_pump | digital | memory | io_pad | esd | bias_generator | matched_pair_cell | unknown",
      "confidence": "high | medium | low",
      "evidence": "one sentence citing specific name tokens, children, or instance counts",
      "alternatives": ["other roles considered with brief reason, may be empty"]
    }}
  ],
  "summary": {{
    "bandgap_cells": ["names of cells classified as bandgap_reference"],
    "ldo_cells": ["names of cells classified as LDO"],
    "oscillator_cells": ["names of cells classified as oscillator"],
    "matched_pair_cells": ["names of cells with high-instance-count matched-pair children"],
    "notes": "any caveats or ambiguities in one sentence"
  }}
}}
"""


def build_2b_input(extraction, semantics_2a):
    chip = extraction["chip"]
    cells = []
    for c in extraction.get("cells", [])[:40]:
        cells.append({
            "name": c["name"],
            "own_polys": c.get("own_polys", 0),
            "deep_polys": c.get("deep_polys", 0),
            "n_instances": c.get("n_instances", 0),
            "n_children": c.get("n_children", 0),
            "n_parents": c.get("n_parents", 0),
            "children": c.get("children", [])[:15],
            "parents": c.get("parents", [])[:8],
            "is_top": c.get("is_top", False),
        })

    # Slim 2A down to just the summary + role classification for context
    slim_2a = {
        "summary": (semantics_2a or {}).get("summary", {}),
        "layers": [
            {"id": l.get("id"), "role": l.get("role"),
             "confidence": l.get("confidence")}
            for l in (semantics_2a or {}).get("layers", [])
        ],
    }
    return chip, cells, slim_2a


def run_section_2b(extraction, semantics_2a, base_url, model):
    chip, cells, slim_2a = build_2b_input(extraction, semantics_2a)
    user_prompt = SECTION_2B_USER_TEMPLATE.format(
        n_cells=len(cells),
        chip=json.dumps(chip, indent=2),
        semantics_2a=json.dumps(slim_2a, indent=2),
        cells=json.dumps(cells, indent=2),
    )

    n_tokens_approx = len(user_prompt) // 4
    print(f"  [2B] prompt: {len(user_prompt):,} chars "
          f"(~{n_tokens_approx:,} tokens)")
    t0 = time.time()
    result, raw = call_llm_json(SECTION_2B_SYSTEM, user_prompt, base_url, model)
    dt = time.time() - t0
    print(f"  [2B] LLM returned in {dt:.1f}s")
    if result is None:
        return {"error": "LLM did not return valid JSON", "raw": raw[:2000]}
    return result


# ── Section 2C: issue-area candidates ───────────────────────────────────

SECTION_2C_SYSTEM = """You are an IC layout reliability analyst. You propose aggressor->victim layer pairs worth deterministic checking in stage 3, targeting specific functional blocks.

Known analog mechanism patterns (use these as templates):
1. rdl_over_bjt_pair (CRITICAL): RDL crossing matched BJT pair (Q1/Q2) creates thermal asymmetry and piezo-stress, skewing Vbe and breaking the bandgap reference.
2. rdl_over_diff (HIGH): RDL over active diffusion induces piezo-stress shifts in threshold/beta.
3. rdl_over_poly (MEDIUM): RDL over poly resistor strings causes systematic offset; matched dividers especially vulnerable.
4. thick_metal_over_poly (CRITICAL): Thick high-current top metal over precision poly resistors heats them asymmetrically; LDO feedback-divider mismatch shifts output.
5. poly_fill_over_input_diff (HIGH): Dummy poly fill over one matched-pair member creates asymmetry; comparator input-referred offset.
6. digital_signal_near_hi_z (HIGH): Switching digital metal parallel to hi-Z oscillator tank node injects noise via parasitic capacitance.

Rules:
- Propose triples only when there's POSITIVE evidence from 2A (layer roles), 2B (cell roles), and layer-pair overlaps.
- Each triple must name a concrete aggressor layer, victim layer, target cells (the blocks where the pattern matters), mechanism (one of the six above or a close variant), and severity.
- If no overlap evidence supports a pattern, SKIP it. Do not hallucinate coverage.
- Output MUST be a single JSON object, no prose before or after, no markdown fences.
"""

SECTION_2C_USER_TEMPLATE = """Propose aggressor->victim check candidates for stage 3.

# CHIP
{chip}

# SECTION 2A LAYER SEMANTICS
{semantics_2a}

# SECTION 2B CELL ROLES
{semantics_2b}

# LAYER-PAIR OVERLAPS (top by intersection area)
{overlaps}

# MATCHED-PAIR CANDIDATES (cells with n_instances >= 2)
{matched_pairs}

Respond with a single JSON object in this exact shape:

{{
  "candidates": [
    {{
      "mechanism": "rdl_over_bjt_pair | rdl_over_diff | rdl_over_poly | thick_metal_over_poly | poly_fill_over_input_diff | digital_signal_near_hi_z | other",
      "aggressor_layer": "LAYER/DATATYPE",
      "victim_layer": "LAYER/DATATYPE",
      "target_cells": ["cell names where this pattern should be checked"],
      "severity": "critical | high | medium | low",
      "supporting_overlap_um2": "numeric or null, from the overlaps input if available",
      "reasoning": "one-to-two sentence citation of 2A layer roles, 2B cell roles, and overlap evidence"
    }}
  ],
  "summary": {{
    "n_candidates": 0,
    "critical_candidates": 0,
    "notes": "any caveats or coverage gaps in one sentence"
  }}
}}
"""


def build_2c_input(extraction, semantics_2a, semantics_2b):
    chip = extraction["chip"]
    overlaps = extraction.get("layer_pair_overlaps", [])[:25]
    matched_pairs = [
        {"name": c["name"],
         "n_instances": c.get("n_instances", 0),
         "deep_polys": c.get("deep_polys", 0),
         "parents": c.get("parents", [])[:5]}
        for c in extraction.get("cells", [])
        if c.get("n_instances", 0) >= 2
    ][:30]

    slim_2a = {
        "summary": (semantics_2a or {}).get("summary", {}),
        "layers": [
            {"id": l.get("id"), "role": l.get("role"),
             "confidence": l.get("confidence")}
            for l in (semantics_2a or {}).get("layers", [])
        ],
    }
    slim_2b = {
        "summary": (semantics_2b or {}).get("summary", {}),
        "cells": [
            {"name": c.get("name"), "role": c.get("role"),
             "confidence": c.get("confidence")}
            for c in (semantics_2b or {}).get("cells", [])
        ],
    }
    return chip, slim_2a, slim_2b, overlaps, matched_pairs


def run_section_2c(extraction, semantics_2a, semantics_2b, base_url, model):
    chip, slim_2a, slim_2b, overlaps, matched_pairs = build_2c_input(
        extraction, semantics_2a, semantics_2b)
    user_prompt = SECTION_2C_USER_TEMPLATE.format(
        chip=json.dumps(chip, indent=2),
        semantics_2a=json.dumps(slim_2a, indent=2),
        semantics_2b=json.dumps(slim_2b, indent=2),
        overlaps=json.dumps(overlaps, indent=2),
        matched_pairs=json.dumps(matched_pairs, indent=2),
    )

    n_tokens_approx = len(user_prompt) // 4
    print(f"  [2C] prompt: {len(user_prompt):,} chars "
          f"(~{n_tokens_approx:,} tokens)")
    t0 = time.time()
    result, raw = call_llm_json(SECTION_2C_SYSTEM, user_prompt, base_url, model)
    dt = time.time() - t0
    print(f"  [2C] LLM returned in {dt:.1f}s")
    if result is None:
        return {"error": "LLM did not return valid JSON", "raw": raw[:2000]}
    return result


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("extraction_json")
    ap.add_argument("--output", default=None,
                    help="Default: semantics.json next to extraction.json")
    ap.add_argument("--sections", default="2a,2b,2c",
                    help="Comma-separated list (2a,2b,2c). Default: full chain.")
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
        print("\n=== Section 2B: cell functional roles ===")
        sem_2a = semantics.get("2a_layers")
        if not sem_2a or "error" in sem_2a:
            print("  (skipped — 2A not available or errored)")
        else:
            semantics["2b_cells"] = run_section_2b(
                extraction, sem_2a, args.llm_url, model)
            with open(out_path, "w") as f:
                json.dump(semantics, f, indent=2)
            print(f"  -> wrote {out_path}")

    if "2c" in sections:
        print("\n=== Section 2C: issue-area candidates ===")
        sem_2a = semantics.get("2a_layers")
        sem_2b = semantics.get("2b_cells")
        if (not sem_2a or "error" in sem_2a
                or not sem_2b or "error" in sem_2b):
            print("  (skipped — 2A or 2B not available or errored)")
        else:
            semantics["2c_candidates"] = run_section_2c(
                extraction, sem_2a, sem_2b, args.llm_url, model)
            with open(out_path, "w") as f:
                json.dump(semantics, f, indent=2)
            print(f"  -> wrote {out_path}")

    print(f"\nDone. Semantics so far: {list(semantics.keys())}")


if __name__ == "__main__":
    main()
