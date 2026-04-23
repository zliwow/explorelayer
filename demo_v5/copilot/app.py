"""
app.py — Streamlit copilot for the v5 GDS review pipeline.

Run:
    pip install -r demo_v5/copilot/requirements.txt
    streamlit run demo_v5/copilot/app.py

Expects extraction.json, semantics.json, issues.json to live in the same
folder. Default pointer is `demo_v5/` next to this app.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb import (  # noqa: E402
    load_kb, layer_role_lookup, cell_role_lookup, build_chat_context,
)
from qwen import discover_model, chat  # noqa: E402
from viz_3d import build_figure  # noqa: E402


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GDS Copilot — v5",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Artifact loading (cached) ──────────────────────────────────────────
@st.cache_data(show_spinner="Loading extraction + semantics + issues...")
def cached_load(extraction_path, mtime):
    # mtime busts the cache when the file changes on disk
    _ = mtime
    return load_kb(extraction_path)


def resolve_kb_path():
    """Sidebar widgets that choose the extraction file to load."""
    st.sidebar.header("Knowledge base")
    default_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_dir = st.sidebar.text_input(
        "Folder with extraction*.json", default_dir,
        help="Folder containing extraction.json, semantics.json, issues.json",
    )
    if not os.path.isdir(kb_dir):
        st.sidebar.error(f"Not a folder: {kb_dir}")
        return None
    candidates = sorted([f for f in os.listdir(kb_dir)
                         if f.startswith("extraction") and f.endswith(".json")])
    if not candidates:
        st.sidebar.error("No extraction*.json found.")
        return None
    picked = st.sidebar.selectbox("Extraction file", candidates)
    return os.path.join(kb_dir, picked)


extraction_path = resolve_kb_path()
if not extraction_path or not os.path.exists(extraction_path):
    st.title("GDS Copilot")
    st.warning("Pick a valid extraction.json in the sidebar.")
    st.stop()

kb = cached_load(extraction_path, os.path.getmtime(extraction_path))
layer_roles = layer_role_lookup(kb)
cell_roles = cell_role_lookup(kb)

chip = kb["extraction"]["chip"]
issues = kb.get("issues") or {}
semantics = kb.get("semantics") or {}
summary = issues.get("summary", {})


# ── Sidebar filters ────────────────────────────────────────────────────
st.sidebar.header("View filters")
severity_filter = st.sidebar.multiselect(
    "Severity",
    ["critical", "high", "medium", "low"],
    default=[s for s in ["critical", "high"] if summary.get(s, 0) > 0] or ["critical", "high"],
)
max_hits = st.sidebar.slider("Max hits to render", 50, 2000, 500, 50)
show_cells = st.sidebar.checkbox("Show cell boxes", value=True)
show_hits = st.sidebar.checkbox("Show hit markers", value=True)

st.sidebar.header("LLM")
llm_url = st.sidebar.text_input("Qwen URL", "http://localhost:8000")
if "qwen_model" not in st.session_state:
    st.session_state.qwen_model = None
if st.sidebar.button("Reconnect / discover model"):
    st.session_state.qwen_model = discover_model(llm_url)
if st.session_state.qwen_model is None:
    st.session_state.qwen_model = discover_model(llm_url)
st.sidebar.caption(f"Model: `{st.session_state.qwen_model or '(not discovered)'}`")


# ── Header + metrics ───────────────────────────────────────────────────
st.title("GDS Copilot")
st.caption(
    f"**{chip.get('top_cell', '?')}** · "
    f"{kb['paths']['extraction']} · "
    f"semantics: {'✓' if semantics else '✗'} · "
    f"issues: {'✓' if issues else '✗'}"
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Die (µm)", f"{chip.get('die_w', 0):.0f} × {chip.get('die_h', 0):.0f}")
m2.metric("Layers", len(kb["extraction"].get("layers", [])))
m3.metric("Top cells", len(kb["extraction"].get("cells", [])))
m4.metric("Candidates (2C)", len((semantics.get("2c_candidates") or {}).get("candidates", [])))
m5.metric("Critical", summary.get("critical", 0))
m6.metric("High", summary.get("high", 0))


# ── Main layout ────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

# ── Left: 3D view + hits table ─────────────────────────────────────────
with left:
    st.subheader("3D chip view")

    highlight_bbox = st.session_state.get("highlight_bbox")
    fig = build_figure(
        kb, layer_roles, cell_roles,
        filter_severity=severity_filter,
        max_hits=max_hits,
        show_cells=show_cells,
        show_hits=show_hits,
        highlight_bbox=highlight_bbox,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "X/Y in µm. Z is a visual stacking convention based on layer role "
        "(RDL on top, diffusion at bottom) — not real process thickness."
    )

    # ── Hits table ─────────────────────────────────────────────────────
    all_hits = issues.get("hits", [])
    filt_hits = [h for h in all_hits if h.get("severity") in severity_filter]
    st.subheader(f"Hits ({len(filt_hits)}/{len(all_hits)} shown)")
    if filt_hits:
        df = pd.DataFrame([{
            "sev": h.get("severity"),
            "mechanism": h.get("mechanism"),
            "target_cell": h.get("target_cell"),
            "aggressor": h.get("aggressor_layer"),
            "victim": h.get("victim_layer"),
            "area_um2": round(h.get("overlap_area_um2", 0), 2),
            "path": " / ".join(h.get("cell_path") or []),
            "bbox": h.get("overlap_bbox"),
        } for h in filt_hits])
        df = df.sort_values(
            by=["sev", "area_um2"],
            key=lambda col: col.map({"critical": 0, "high": 1, "medium": 2, "low": 3}).fillna(9)
                            if col.name == "sev" else -col,
            ascending=[True, True],
        )
        event = st.dataframe(
            df, use_container_width=True, height=320,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "area_um2": st.column_config.NumberColumn("area µm²", format="%.2f"),
                "bbox": st.column_config.ListColumn("bbox"),
            },
        )
        sel = (event.selection.rows if event and getattr(event, "selection", None) else [])
        if sel:
            row = df.iloc[sel[0]]
            st.session_state.highlight_bbox = row["bbox"]
            st.session_state.selected_hit = row.to_dict()
            st.info(
                f"Selected: **{row['mechanism']}** [{row['sev']}] at "
                f"`{row['target_cell']}` — area {row['area_um2']} µm². "
                f"(bbox highlighted in 3D — scroll up to see it)"
            )
    else:
        st.info("No hits match current filter.")


# ── Right: chat ────────────────────────────────────────────────────────
with right:
    st.subheader("Ask about this chip")
    st.caption("Copilot answers using the preloaded stage 1+2+3 facts.")

    # Seed messages from a selected hit if the user clicked "Explain this"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat"):
        st.session_state.messages = []
    sel_hit = st.session_state.get("selected_hit")
    if sel_hit and col_b.button("Explain selected hit"):
        q = (f"Explain this hit: {sel_hit['mechanism']} ({sel_hit['sev']}) "
             f"at cell {sel_hit['target_cell']}, aggressor "
             f"{sel_hit['aggressor']} over victim {sel_hit['victim']}, "
             f"overlap area {sel_hit['area_um2']} µm². "
             f"Why is this a concern, and what should a reviewer check?")
        st.session_state.messages.append({"role": "user", "content": q})

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("e.g. 'which layer is RDL?' or 'summarize the bandgap risks'")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # If the last message is from the user and not yet answered, call Qwen
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "user":
        model = st.session_state.qwen_model
        if not model:
            with st.chat_message("assistant"):
                st.error(f"No model discovered at {llm_url}. Check the sidebar.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"(no model at {llm_url})",
            })
            st.stop()

        system_msg = (
            "You are an IC layout review copilot for a GDS chip layout. "
            "Answer the user's question using ONLY the facts below. When you "
            "state a fact, cite the source in-line like "
            "`[semantics.2a_layers: 50/0 = RDL]` or `[issues.hits[3]]`. "
            "If the answer isn't in the facts, say so — do NOT guess.\n\n"
            + build_chat_context(kb)
        )
        api_msgs = [{"role": "system", "content": system_msg}]
        # Keep last 8 turns of chat history (no more — the system blob is big)
        api_msgs.extend(msgs[-8:])

        with st.chat_message("assistant"):
            with st.spinner("Qwen is thinking..."):
                try:
                    answer = chat(api_msgs, llm_url, model)
                except Exception as e:
                    answer = f"LLM error: `{e}`"
            st.markdown(answer or "(empty response)")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer or "(empty response)",
        })


# ── Footer: show paths for debugging ───────────────────────────────────
with st.expander("Artifact paths", expanded=False):
    for k, v in kb["paths"].items():
        st.code(f"{k}: {v or '(missing)'}")
