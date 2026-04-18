#!/usr/bin/env python3
"""
Generate a synthetic but complex analog IP GDS with FIVE distinct
analog blocks and FOUR different planted layout issues plus ONE
benign-looking pattern that a naive detector will flag as a false
positive.

This is the ground-truth harness for the Claude-vs-Qwen comparison.
A good reviewer has to:
  - write specific prose for each of four different failure mechanisms
    (RDL-over-matched-pair, current metal over precision resistors,
    asymmetric dummy fill on a matched diff pair, unshielded digital
    signal near a high-Z oscillator node)
  - AND recognise that the fifth flagged finding (RDL landing on an
    ESD clamp pad) is NOT a real issue — it's the intended bond-pad
    routing.

Layer stack (sky130-ish, same numbering as demo_synthetic/ so the
rendering styles carry over):
    1/0   diff          active area / source-drain diffusion
    2/0   nwell
    5/0   poly
    6/0   poly_fill     dummy poly fill (the planted asymmetry)
   10/0   contact
   11/0   met1
   22/0   met2
   33/0   met3          top metal signal routing
   40/0   met_thick     thick top/analog metal (LDO pass device rail)
   50/0   rdl           redistribution / package top
   60/0   pad           bond pad opening
   82/5   bjt_marker    sky130-style PNP marker
   90/0   hi_z_marker   annotates the sensitive high-Z node
  200/0   label
"""

import gdstk

LAYERS = {
    "diff":         (1, 0),
    "nwell":        (2, 0),
    "poly":         (5, 0),
    "poly_fill":    (6, 0),
    "contact":     (10, 0),
    "met1":        (11, 0),
    "met2":        (22, 0),
    "met3":        (33, 0),
    "met_thick":   (40, 0),
    "rdl":         (50, 0),
    "pad":         (60, 0),
    "bjt_marker":  (82, 5),
    "hi_z_marker": (90, 0),
    "label":      (200, 0),
}


def rect(cell, x, y, w, h, layer_key, datatype=None):
    l, dt = LAYERS[layer_key]
    if datatype is not None:
        dt = datatype
    cell.add(gdstk.rectangle((x, y), (x + w, y + h), layer=l, datatype=dt))


def add_label(cell, text, x, y):
    l, dt = LAYERS["label"]
    cell.add(gdstk.Label(text, (x, y), layer=l, texttype=dt))


# ── Leaf device footprints ────────────────────────────────────────────

def bjt_pnp_cell():
    c = gdstk.Cell("BJT_PNP")
    rect(c, 0, 0, 30, 30, "nwell")
    rect(c, 3, 3, 24, 24, "diff")
    rect(c, 12, 12, 6, 6, "diff")
    rect(c, 13, 13, 4, 4, "contact")
    rect(c, 13, 13, 4, 4, "met1")
    for (cx, cy) in [(6, 6), (22, 6), (6, 22), (22, 22)]:
        rect(c, cx, cy, 2, 2, "contact")
    rect(c, 0, 0, 30, 30, "bjt_marker")
    return c


def nmos_cell(name, w=20, h=15):
    c = gdstk.Cell(name)
    rect(c, 0, 0, w, h, "diff")
    rect(c, w / 2 - 1, -2, 2, h + 4, "poly")
    rect(c, 2, h / 2 - 1, 2, 2, "contact")
    rect(c, w - 4, h / 2 - 1, 2, 2, "contact")
    rect(c, 0, 0, w / 2 - 1, h, "met1")
    rect(c, w / 2 + 1, 0, w / 2 - 1, h, "met1")
    return c


def pmos_big_cell():
    c = gdstk.Cell("PMOS_PASS")
    rect(c, 0, 0, 80, 40, "nwell")
    rect(c, 2, 2, 76, 36, "diff")
    for i in range(4):
        rect(c, 8 + i * 20, -2, 3, 44, "poly")
    for i in range(4):
        rect(c, 4 + i * 20, 18, 3, 4, "contact")
    rect(c, 0, 0, 80, 40, "met1")
    return c


def poly_resistor_cell(length=60, width=4):
    c = gdstk.Cell(f"RES_POLY_{int(length)}")
    rect(c, 0, 0, length, width, "poly")
    rect(c, 1, width / 2 - 1, 2, 2, "contact")
    rect(c, length - 3, width / 2 - 1, 2, 2, "contact")
    rect(c, 0, 0, 5, width, "met1")
    rect(c, length - 5, 0, 5, width, "met1")
    return c


def esd_clamp_cell():
    c = gdstk.Cell("ESD_CLAMP_NMOS")
    rect(c, 0, 0, 100, 60, "diff")
    for i in range(10):
        rect(c, 4 + i * 10, -2, 3, 64, "poly")
    for i in range(11):
        rect(c, 2 + i * 10, 28, 3, 4, "contact")
    rect(c, 0, 0, 100, 60, "met1")
    return c


def inductor_like_cell():
    c = gdstk.Cell("IND_POLY")
    # Meander "inductor" made of poly — decorative; what matters is its
    # HI_Z marker for the checker.
    x = 0
    y = 0
    turns = 4
    pitch = 8
    length = 120
    for t in range(turns):
        rect(c, x, y + t * pitch, length, 3, "poly")
        if t < turns - 1:
            rect(c, length - 3, y + t * pitch, 3, pitch, "poly")
    rect(c, 0, 0, 2, turns * pitch, "met1")
    rect(c, length - 2, 0, 2, turns * pitch, "met1")
    # High-Z marker — annotates the tank node
    rect(c, 0, 0, length, turns * pitch, "hi_z_marker")
    return c


def pad_cell():
    c = gdstk.Cell("PAD")
    rect(c, 0, 0, 80, 80, "met3")
    rect(c, 10, 10, 60, 60, "pad")
    rect(c, 0, 0, 80, 80, "rdl")
    return c


# ── Composite blocks ───────────────────────────────────────────────────

def build_bandgap():
    c = gdstk.Cell("BANDGAP_REF")
    bjt = bjt_pnp_cell()
    res = poly_resistor_cell(length=60)
    # Matched pair Q1/Q2, 40 µm apart
    c.add(gdstk.Reference(bjt, origin=(20, 20)))
    add_label(c, "Q1", 35, 52)
    c.add(gdstk.Reference(bjt, origin=(70, 20)))
    add_label(c, "Q2", 85, 52)
    # Resistor string
    for i in range(3):
        c.add(gdstk.Reference(res, origin=(20 + i * 55, 80)))
    add_label(c, "R_ref", 50, 90)
    # Block outline
    c.add(gdstk.rectangle((0, 0), (200, 120),
                          layer=LAYERS["met3"][0], datatype=1))
    add_label(c, "BANDGAP_REF", 80, 110)
    return c


def build_ldo():
    """LDO: pass PMOS + feedback divider (two matched poly resistors).
    Planted issue: a thick current-carrying top-metal rail will be
    routed in the top cell directly over the feedback divider."""
    c = gdstk.Cell("LDO_REG")
    c.add(gdstk.Reference(pmos_big_cell(), origin=(20, 70)))
    add_label(c, "M_pass", 40, 115)
    # Feedback divider — two matched poly resistors at the BOTTOM of
    # the block (this is the region the top metal will cover).
    r = poly_resistor_cell(length=80, width=4)
    c.add(gdstk.Reference(r, origin=(30, 20)))
    add_label(c, "R_fb_top", 50, 28)
    c.add(gdstk.Reference(r, origin=(30, 35)))
    add_label(c, "R_fb_bot", 50, 43)
    c.add(gdstk.rectangle((0, 0), (200, 150),
                          layer=LAYERS["met3"][0], datatype=1))
    add_label(c, "LDO_REG", 80, 140)
    return c


def build_comp():
    """Comparator with matched input NMOS pair.
    Planted issue: poly dummy fill is placed on ONE side of the pair
    only, breaking matching symmetry."""
    c = gdstk.Cell("COMP_INPUT")
    nm = nmos_cell("MOS_NMOS", w=24, h=16)
    # Matched pair Mi_p / Mi_n 30 um apart
    c.add(gdstk.Reference(nm, origin=(20, 30)))
    add_label(c, "Mi_p", 32, 50)
    c.add(gdstk.Reference(nm, origin=(80, 30)))
    add_label(c, "Mi_n", 92, 50)
    # Tail current source below
    c.add(gdstk.Reference(nm, origin=(50, 10)))
    add_label(c, "Mtail", 62, 12)
    # Asymmetric dummy poly fill — placed across Mi_p (left input) ONLY,
    # nothing on Mi_n. Overlaps Mi_p's diff so the overlap detector catches
    # it; the matching reviewer must explain *why* one-sided fill breaks
    # the input pair.
    for i in range(3):
        rect(c, 22 + i * 5, 32, 3, 12, "poly_fill")
    c.add(gdstk.rectangle((0, 0), (180, 80),
                          layer=LAYERS["met3"][0], datatype=1))
    add_label(c, "COMP_INPUT", 70, 70)
    return c


def build_osc():
    """LC-ish oscillator with a high-Z tank node.
    Planted issue: we will route a switching digital M2 signal in the
    top cell running parallel to the tank node, with no ground shield
    in between."""
    c = gdstk.Cell("OSC_LC")
    c.add(gdstk.Reference(inductor_like_cell(), origin=(30, 40)))
    add_label(c, "L_tank", 50, 80)
    # Output buffer inverter
    c.add(gdstk.Reference(nmos_cell("MOS_NMOS", w=20, h=15), origin=(170, 20)))
    add_label(c, "M_buf", 180, 30)
    c.add(gdstk.rectangle((0, 0), (200, 150),
                          layer=LAYERS["met3"][0], datatype=1))
    add_label(c, "OSC_LC", 80, 140)
    return c


def build_esd():
    """ESD IO block: clamp + pad. Metal routing from pad to clamp is
    EXPECTED — the detector will flag it, and a good reviewer should
    classify it as a false positive."""
    c = gdstk.Cell("ESD_IO")
    c.add(gdstk.Reference(esd_clamp_cell(), origin=(20, 20)))
    add_label(c, "M_clamp", 60, 15)
    c.add(gdstk.rectangle((0, 0), (150, 100),
                          layer=LAYERS["met3"][0], datatype=1))
    add_label(c, "ESD_IO", 60, 90)
    return c


# ── Top cell ───────────────────────────────────────────────────────────

def build_chip():
    lib = gdstk.Library(unit=1e-6, precision=1e-9)

    blocks = {
        "bg":   build_bandgap(),
        "ldo":  build_ldo(),
        "comp": build_comp(),
        "osc":  build_osc(),
        "esd":  build_esd(),
        "pad":  pad_cell(),
    }
    for c in blocks.values():
        lib.add(c)

    # Collect all referenced subcells
    seen = {c.name for c in lib.cells}
    to_add = []
    for c in list(lib.cells):
        for ref in c.references:
            rc = ref.cell
            if isinstance(rc, gdstk.Cell) and rc.name not in seen:
                seen.add(rc.name)
                to_add.append(rc)
    while to_add:
        c = to_add.pop()
        lib.add(c)
        for ref in c.references:
            rc = ref.cell
            if isinstance(rc, gdstk.Cell) and rc.name not in seen:
                seen.add(rc.name)
                to_add.append(rc)

    top = gdstk.Cell("analog_ip_top")
    lib.add(top)

    # Block placements (die ~1200 x 800)
    bg_pos   = (80,  120)
    ldo_pos  = (350, 100)
    comp_pos = (700,  80)
    osc_pos  = (950, 120)
    esd_pos  = (80,  550)

    top.add(gdstk.Reference(blocks["bg"],   origin=bg_pos))
    top.add(gdstk.Reference(blocks["ldo"],  origin=ldo_pos))
    top.add(gdstk.Reference(blocks["comp"], origin=comp_pos))
    top.add(gdstk.Reference(blocks["osc"],  origin=osc_pos))
    top.add(gdstk.Reference(blocks["esd"],  origin=esd_pos))

    # ── Planted issue #1: RDL over BANDGAP matched pair ───────────────
    # BANDGAP at (80,120); Q1 at local (20,20)->(100,140), Q2 at (150,140)
    # RDL strip at y=135..155 across x=70..270 covers BOTH devices.
    rect(top, 70, 135, 210, 20, "rdl")
    add_label(top, "RDL_VIN", 170, 145)

    # ── Planted issue #2: thick top-metal rail over LDO FB divider ────
    # LDO at (350,100); FB resistors at local y=20..45 -> global y=120..145
    # Route a thick (wide) metal carrying the LDO output current
    # directly across the divider.
    rect(top, 340, 120, 220, 30, "met_thick")
    add_label(top, "M_LDO_out", 450, 135)

    # ── Planted issue #3: already in COMP block (asymmetric poly fill) ─
    # Nothing to add at top level; fault lives inside COMP_INPUT itself.

    # ── Planted issue #4: digital signal parallel to OSC hi-Z node ────
    # OSC at (950,120); inductor at local (30,40)->(980,160) size 120x32
    # Run a long met2 digital line straight across the tank node with no
    # ground shield in between — capacitive injection into a high-Q LC.
    rect(top, 970, 170, 125, 4, "met2")
    add_label(top, "CLK_DIG", 1025, 147)

    # ── Planted NON-issue #5: RDL from pad to ESD clamp (expected) ────
    # Add a pad at top-left, then route RDL down to the ESD clamp.
    top.add(gdstk.Reference(blocks["pad"], origin=(60, 680)))
    add_label(top, "PAD_VDD", 100, 720)
    # RDL strap from pad down to ESD
    rect(top, 85, 570, 35, 120, "rdl")

    # A couple more pads for realism
    top.add(gdstk.Reference(blocks["pad"], origin=(300, 680)))
    add_label(top, "PAD_GND", 340, 720)
    top.add(gdstk.Reference(blocks["pad"], origin=(540, 680)))
    add_label(top, "PAD_VIN", 580, 720)
    top.add(gdstk.Reference(blocks["pad"], origin=(780, 680)))
    add_label(top, "PAD_VOUT", 820, 720)
    top.add(gdstk.Reference(blocks["pad"], origin=(1020, 680)))
    add_label(top, "PAD_CLK", 1060, 720)

    # Die outline
    rect(top, 0, 0, 1200, 800, "met3", datatype=1)
    add_label(top, "analog_ip_top", 600, 780)
    return lib


def write_cdl(path):
    """Hand-written matching netlist — short but enough for evidence
    panels to quote."""
    body = [
        "* complex_chip.cdl — matching netlist for the complex demo",
        "",
        ".SUBCKT BANDGAP_REF vbg vdda vssa",
        "+ XQ1 vssa bg_base bg_e1 BJT_PNP   $ matched pair",
        "+ XQ2 vssa bg_base bg_e2 BJT_PNP   $ matched pair",
        "+ XR1 bg_e2 bg_res_a RES_POLY",
        "+ XR2 bg_res_a bg_res_b RES_POLY",
        "+ XR3 bg_res_b vssa RES_POLY",
        ".ENDS",
        "",
        ".SUBCKT LDO_REG vout vref vfb vdda vssa",
        "+ XM_pass vout vg_pass vdda vdda PMOS_PASS",
        "+ XR_fb_top vout vfb RES_POLY   $ precision divider, matched",
        "+ XR_fb_bot vfb vssa RES_POLY   $ precision divider, matched",
        ".ENDS",
        "",
        ".SUBCKT COMP_INPUT inp inn out vdda vssa",
        "+ XMi_p inp vgate n_src vssa MOS_NMOS   $ matched input pair",
        "+ XMi_n inn vgate n_src vssa MOS_NMOS   $ matched input pair",
        "+ XMtail n_src vbias vssa vssa MOS_NMOS",
        ".ENDS",
        "",
        ".SUBCKT OSC_LC vtank vout vdda vssa",
        "+ XL_tank vtank vdda IND_POLY   $ high-Z tank node",
        "+ XM_buf  vtank vout vdda vssa MOS_NMOS",
        ".ENDS",
        "",
        ".SUBCKT ESD_IO pad core vssa",
        "+ XM_clamp pad vssa vssa vssa ESD_CLAMP_NMOS",
        ".ENDS",
        "",
        ".END",
    ]
    with open(path, "w") as f:
        f.write("\n".join(body) + "\n")


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    gds_path = os.path.join(here, "complex_chip.gds")
    cdl_path = os.path.join(here, "complex_chip.cdl")

    lib = build_chip()
    lib.write_gds(gds_path)
    print(f"Wrote {gds_path}  ({sum(len(c.polygons) for c in lib.cells)} leaf polygons)")

    write_cdl(cdl_path)
    print(f"Wrote {cdl_path}")


if __name__ == "__main__":
    main()
