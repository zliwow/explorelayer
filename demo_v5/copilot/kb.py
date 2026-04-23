"""
kb.py — load and index the three stage artifacts for the Streamlit copilot.

extraction.json + semantics.json + issues.json form the knowledge base.
Stage 2/3 write with fixed filenames next to extraction.json, so we look
there first, then fall back to the user's explicit paths.
"""

import json
import os


ARTIFACT_NAMES = {
    "semantics": ["semantics.json"],
    "issues":    ["issues.json"],
}


def _sibling(path, names):
    """Return the first existing sibling filename, or None."""
    d = os.path.dirname(os.path.abspath(path))
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def load_kb(extraction_path):
    """
    Load the three artifacts. Semantics and issues are optional so the app
    still renders on a fresh chip where only stage 1 has been run.
    """
    extraction = json.load(open(extraction_path))

    sem_path = _sibling(extraction_path, ARTIFACT_NAMES["semantics"])
    iss_path = _sibling(extraction_path, ARTIFACT_NAMES["issues"])

    return {
        "extraction": extraction,
        "semantics": json.load(open(sem_path)) if sem_path else None,
        "issues":    json.load(open(iss_path)) if iss_path else None,
        "paths": {
            "extraction": os.path.abspath(extraction_path),
            "semantics":  sem_path,
            "issues":     iss_path,
        },
    }


# ── Lookups ─────────────────────────────────────────────────────────────

def layer_role_lookup(kb):
    """{ 'LAYER/DATATYPE': role_string } from semantics 2A, {} if absent."""
    sem = kb.get("semantics") or {}
    section = sem.get("2a_layers") or {}
    return {l.get("id"): l.get("role") for l in section.get("layers", [])
            if l.get("id")}


def cell_role_lookup(kb):
    """{ cell_name: role_string } from semantics 2B, {} if absent."""
    sem = kb.get("semantics") or {}
    section = sem.get("2b_cells") or {}
    return {c.get("name"): c.get("role") for c in section.get("cells", [])
            if c.get("name")}


def layer_evidence(kb, layer_id):
    sem = kb.get("semantics") or {}
    section = sem.get("2a_layers") or {}
    for l in section.get("layers", []):
        if l.get("id") == layer_id:
            return l
    return None


# ── Chat context builder ────────────────────────────────────────────────

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}


def build_chat_context(kb, top_hits=30):
    """
    Flatten the KB into a compact text blob for Qwen's system prompt.
    We'd rather keep this small than exhaustive — the point is to give
    the model grounding, not dump every number.
    """
    lines = []
    chip = kb["extraction"]["chip"]
    lines.append(f"=== CHIP ===")
    lines.append(f"top_cell: {chip.get('top_cell')}")
    lines.append(f"die: {chip.get('die_w', 0):.1f} x {chip.get('die_h', 0):.1f} um "
                 f"(area {chip.get('die_area', 0):.0f} um^2)")
    lines.append(f"hierarchy depth: {chip.get('hierarchy_max_depth')}")
    lines.append(f"total polygons: {chip.get('total_polygons'):,}")
    lines.append(f"cells in library: {chip.get('n_cells_in_library'):,}")

    sem = kb.get("semantics") or {}
    iss = kb.get("issues") or {}

    # 2A layer roles
    layers_2a = (sem.get("2a_layers") or {}).get("layers", [])
    if layers_2a:
        lines.append("")
        lines.append("=== LAYER ROLES (2A) ===")
        for l in layers_2a:
            lines.append(
                f"  {l.get('id')}: role={l.get('role')} "
                f"confidence={l.get('confidence')} | evidence: {l.get('evidence', '')}"
            )
        summ_2a = (sem.get("2a_layers") or {}).get("summary") or {}
        if summ_2a:
            lines.append(f"  summary: {json.dumps(summ_2a)}")

    # 2B cell roles — only those with a definite role
    cells_2b = (sem.get("2b_cells") or {}).get("cells", [])
    if cells_2b:
        lines.append("")
        lines.append("=== CELL ROLES (2B) ===")
        for c in cells_2b[:40]:
            lines.append(
                f"  {c.get('name')}: role={c.get('role')} "
                f"confidence={c.get('confidence')} | evidence: {c.get('evidence', '')}"
            )
        summ_2b = (sem.get("2b_cells") or {}).get("summary") or {}
        if summ_2b:
            lines.append(f"  summary: {json.dumps(summ_2b)}")

    # 2C candidates (the mechanism hypotheses)
    cand_2c = (sem.get("2c_candidates") or {}).get("candidates", [])
    if cand_2c:
        lines.append("")
        lines.append("=== CANDIDATE MECHANISMS (2C) ===")
        for c in cand_2c:
            lines.append(
                f"  {c.get('mechanism')} [{c.get('severity')}]: "
                f"{c.get('aggressor_layer')} -> {c.get('victim_layer')} in "
                f"{c.get('target_cells')} | {c.get('reasoning', '')}"
            )

    # Stage 3 hits
    summ = iss.get("summary") or {}
    if summ:
        lines.append("")
        lines.append("=== ISSUES SUMMARY (stage 3) ===")
        lines.append(
            f"  {summ.get('n_hits', 0)} hits from {summ.get('n_candidates', 0)} candidates; "
            f"critical={summ.get('critical', 0)}, high={summ.get('high', 0)}, "
            f"medium={summ.get('medium', 0)}, low={summ.get('low', 0)}"
        )
        by_mech = summ.get("by_mechanism") or {}
        if by_mech:
            lines.append(f"  by_mechanism: {json.dumps(by_mech)}")

    hits = iss.get("hits") or []
    if hits:
        sorted_hits = sorted(
            hits,
            key=lambda h: (SEVERITY_RANK.get(h.get("severity"), 9),
                           -h.get("overlap_area_um2", 0))
        )[:top_hits]
        lines.append("")
        lines.append(f"=== TOP {len(sorted_hits)} HITS (by severity, then area) ===")
        for h in sorted_hits:
            path = "/".join(h.get("cell_path") or [])
            bb = h.get("overlap_bbox") or []
            lines.append(
                f"  [{h.get('severity')}] {h.get('mechanism')} @ {h.get('target_cell')} "
                f"(path={path}) {h.get('aggressor_layer')}->{h.get('victim_layer')} "
                f"area={h.get('overlap_area_um2', 0):.2f}um^2 bbox={bb}"
            )

    return "\n".join(lines)


def layer_id_to_role_stack(kb):
    """Map layer_id to its role for 3D z-height lookup."""
    return layer_role_lookup(kb)
