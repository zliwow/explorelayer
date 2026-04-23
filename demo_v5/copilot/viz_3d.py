"""
viz_3d.py — Plotly 3D rendering for the chip knowledge base.

Coordinates:
  - X/Y in µm (from the chip's die bbox)
  - Z is a stacking-order convention derived from the layer's semantic
    role (stage 2A), NOT real process thickness. This is a visual aid
    so RDL sits above top metal above poly, etc.

Renderables:
  - die outline                 — transparent wireframe box
  - target-cell instance boxes  — colored by 2B cell role
  - issue hits                  — 3D markers colored by severity, hoverable
"""

import plotly.graph_objects as go


# ── Z-height convention (visual only; not real process thickness) ──────
ROLE_Z = {
    "bump":                12.0,
    "RDL":                 11.0,
    "passivation":         10.5,
    "pad_opening":         10.0,
    "top_metal":            9.0,
    "seal_ring":            9.5,
    "intermediate_metal":   7.0,
    "via":                  6.0,
    "contact":              5.0,
    "fill":                 4.0,
    "unclassified":         3.0,
    "marker_text":          8.0,
    "capacitor_marker":     2.5,
    "resistor_marker":      2.5,
    "poly":                 2.0,
    "BJT_marker":           1.5,
    "MOS_marker":           1.5,
    "diffusion":            1.0,
    "implant":              1.0,
    "nwell":                0.5,
    "pwell":                0.5,
    "boundary":             0.0,
}

ROLE_COLOR = {
    "bandgap_reference":   "#ff4d4d",
    "LDO":                 "#4d8bff",
    "oscillator":          "#b34dff",
    "buck_converter":      "#4dff8b",
    "boost_converter":     "#4dffb3",
    "charge_pump":         "#ffbb4d",
    "comparator":          "#ffb34d",
    "digital":             "#999999",
    "memory":              "#cccc00",
    "io_pad":              "#4dffff",
    "esd":                 "#ff4dff",
    "bias_generator":      "#8888ff",
    "matched_pair_cell":   "#ff8888",
    "opamp":               "#88ff88",
    "level_shifter":       "#ffaa88",
    "power_switch":        "#aaffaa",
    "chip_top":            "#cccccc",
    "unknown":             "#666666",
}

SEVERITY_COLOR = {
    "critical": "#ff0000",
    "high":     "#ff8800",
    "medium":   "#ffee00",
    "low":      "#00cc66",
    "unknown":  "#888888",
}

SEVERITY_SIZE = {
    "critical": 9,
    "high":     6,
    "medium":   5,
    "low":      4,
    "unknown":  4,
}


def _wireframe_box(x0, y0, x1, y1, z0, z1, color, name, width=2, opacity=1.0):
    """Scatter3d trace that draws the 12 edges of a box."""
    # Bottom rectangle
    xs = [x0, x1, x1, x0, x0, None,
          # Top rectangle
          x0, x1, x1, x0, x0, None,
          # Vertical edges
          x0, x0, None,
          x1, x1, None,
          x1, x1, None,
          x0, x0]
    ys = [y0, y0, y1, y1, y0, None,
          y0, y0, y1, y1, y0, None,
          y0, y0, None,
          y0, y0, None,
          y1, y1, None,
          y1, y1]
    zs = [z0, z0, z0, z0, z0, None,
          z1, z1, z1, z1, z1, None,
          z0, z1, None,
          z0, z1, None,
          z0, z1, None,
          z0, z1]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


# ── Main figure builder ────────────────────────────────────────────────

def build_figure(kb, layer_roles, cell_roles,
                 filter_severity=None, max_hits=500,
                 show_cells=True, show_hits=True,
                 highlight_bbox=None):
    """
    Build the 3D figure. Everything is grounded in world coordinates from
    stage 1's transform-aware hierarchy walk.
    """
    extraction = kb["extraction"]
    issues = kb.get("issues") or {}
    chip = extraction["chip"]

    die = chip.get("die_bbox") or [0, 0, 1, 1]
    dx0, dy0, dx1, dy1 = die
    z_min = min(ROLE_Z.values())
    z_max = max(ROLE_Z.values())

    fig = go.Figure()

    # ── Die wireframe ──────────────────────────────────────────────────
    fig.add_trace(_wireframe_box(
        dx0, dy0, dx1, dy1, z_min, z_max,
        "rgba(180,180,180,0.4)", f"die ({chip.get('top_cell','?')})",
        width=3, opacity=0.6,
    ))

    # ── Cell wireframes (colored by 2B role) ───────────────────────────
    if show_cells:
        for c in extraction.get("cells", []):
            if c.get("is_top"):
                continue
            bb = c.get("bbox_any_instance")
            if not bb or len(bb) != 4:
                continue
            role = cell_roles.get(c["name"], "unknown")
            color = ROLE_COLOR.get(role, ROLE_COLOR["unknown"])
            n_inst = c.get("n_instances", 0)
            fig.add_trace(_wireframe_box(
                bb[0], bb[1], bb[2], bb[3], z_min, z_max,
                color,
                f"{c['name']} [{role}] x{n_inst}",
                width=1 if role == "unknown" else 2,
                opacity=0.5 if role == "unknown" else 0.9,
            ))

    # ── Issue hits (markers) ───────────────────────────────────────────
    if show_hits:
        hits = issues.get("hits") or []
        if filter_severity:
            hits = [h for h in hits if h.get("severity") in filter_severity]
        # Priority: critical first, then largest area
        hits = sorted(hits, key=lambda h: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(h.get("severity"), 4),
            -h.get("overlap_area_um2", 0)
        ))[:max_hits]

        by_sev = {}
        for h in hits:
            by_sev.setdefault(h.get("severity", "unknown"), []).append(h)

        for sev, hlist in by_sev.items():
            xs, ys, zs, texts = [], [], [], []
            for h in hlist:
                bb = h.get("overlap_bbox") or [0, 0, 0, 0]
                cx = (bb[0] + bb[2]) / 2
                cy = (bb[1] + bb[3]) / 2
                role = layer_roles.get(h.get("aggressor_layer"), "unclassified")
                cz = ROLE_Z.get(role, 5.0)
                xs.append(cx); ys.append(cy); zs.append(cz)
                texts.append(
                    f"<b>{h.get('mechanism')}</b> ({sev})<br>"
                    f"cell: {h.get('target_cell')}<br>"
                    f"path: {'/'.join(h.get('cell_path') or [])}<br>"
                    f"{h.get('aggressor_layer')} → {h.get('victim_layer')}<br>"
                    f"area: {h.get('overlap_area_um2', 0):.2f} µm²<br>"
                    f"bbox: [{bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f}]"
                )
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(
                    size=SEVERITY_SIZE.get(sev, 4),
                    color=SEVERITY_COLOR.get(sev, "#888"),
                    opacity=0.85,
                    line=dict(color="black", width=0.5),
                ),
                text=texts, hoverinfo="text",
                name=f"{sev} hits ({len(hlist)})",
            ))

    # ── Optional highlight box (for selected hit) ─────────────────────
    if highlight_bbox and len(highlight_bbox) == 4:
        hb = highlight_bbox
        fig.add_trace(_wireframe_box(
            hb[0], hb[1], hb[2], hb[3], z_min, z_max,
            "#ffff00", "selected hit",
            width=4, opacity=1.0,
        ))

    # ── Scene setup ────────────────────────────────────────────────────
    dw = max(dx1 - dx0, 1.0)
    dh = max(dy1 - dy0, 1.0)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Layer stack (role-based, not real thickness)",
            aspectmode="manual",
            aspectratio=dict(x=2, y=2 * dh / dw, z=0.6),
            bgcolor="#111",
            xaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333"),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        legend=dict(x=0, y=1, bgcolor="rgba(20,20,20,0.6)", font=dict(size=11)),
        paper_bgcolor="#111",
    )
    return fig
