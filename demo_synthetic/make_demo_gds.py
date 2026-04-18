#!/usr/bin/env python3
"""
Generate a synthetic but realistic analog mini-chip GDS with a planted
layout issue: an RDL strip crossing directly over a matched BJT pair
(Q1 / Q2) in the bandgap-reference block.

This is the 'ground truth' for the demo — we know exactly where the
issue is, so we can verify the detection + annotation pipeline.

Layer stack (inspired by SKY130 but simplified):
   1/0   diff       (active area / diffusion)
   5/0   poly
  10/0   contact
  11/0   met1
  22/0   met2
  33/0   met3        (top of normal metal routing)
  50/0   rdl         (redistribution — packaging/top, the 'issue' layer)
  60/0   pad         (bond pad opening)
  82/5   bjt_marker  (PNP/NPN marker — sky130 style)
 200/0   label
"""

import gdstk

LAYERS = {
    "diff":       (1, 0),
    "poly":       (5, 0),
    "contact":    (10, 0),
    "met1":       (11, 0),
    "met2":       (22, 0),
    "met3":       (33, 0),
    "rdl":        (50, 0),
    "pad":        (60, 0),
    "bjt_marker": (82, 5),
    "label":      (200, 0),
}


def rect(cell, x, y, w, h, layer_key, datatype=None):
    """Add a rectangle polygon to cell on the given layer."""
    l, dt = LAYERS[layer_key]
    if datatype is not None:
        dt = datatype
    cell.add(gdstk.rectangle((x, y), (x + w, y + h), layer=l, datatype=dt))


def bjt_cell(name):
    """
    A single PNP BJT footprint:
      - diff ring (base)
      - inner emitter (diff + contact)
      - outer collector ring (met1)
      - bjt_marker layer covering the device extent
    Each BJT is 30 x 30 um.
    """
    c = gdstk.Cell(name)
    # Outer collector well (met1 ring)
    rect(c, 0, 0, 30, 30, "met1")
    # Cut inner hole — do it by drawing base diffusion on top
    rect(c, 3, 3, 24, 24, "diff")
    # Emitter in the center: diff + contact + met1
    rect(c, 12, 12, 6, 6, "diff")
    rect(c, 13, 13, 4, 4, "contact")
    rect(c, 13, 13, 4, 4, "met1")
    # Base contacts (ring of contacts around emitter — simplified as 4 squares)
    for (cx, cy) in [(6, 6), (22, 6), (6, 22), (22, 22)]:
        rect(c, cx, cy, 2, 2, "contact")
    # BJT marker over entire device — this is what the checker looks for
    rect(c, 0, 0, 30, 30, "bjt_marker")
    return c


def mosfet_cell(name, w=20, h=15):
    """Simple MOSFET-like footprint: diff under poly gate, source/drain contacts."""
    c = gdstk.Cell(name)
    # Diff
    rect(c, 0, 0, w, h, "diff")
    # Poly gate bisecting diff
    rect(c, w / 2 - 1, -2, 2, h + 4, "poly")
    # S/D contacts + met1
    rect(c, 2, h / 2 - 1, 2, 2, "contact")
    rect(c, w - 4, h / 2 - 1, 2, 2, "contact")
    rect(c, 0, 0, w / 2 - 1, h, "met1")
    rect(c, w / 2 + 1, 0, w / 2 - 1, h, "met1")
    return c


def resistor_cell(name, length=50, width=4):
    """Poly resistor with contacts on each end."""
    c = gdstk.Cell(name)
    rect(c, 0, 0, length, width, "poly")
    rect(c, 1, width / 2 - 1, 2, 2, "contact")
    rect(c, length - 3, width / 2 - 1, 2, 2, "contact")
    rect(c, 0, 0, 5, width, "met1")
    rect(c, length - 5, 0, 5, width, "met1")
    return c


def pad_cell(name):
    """Simple 80x80 bond pad: met3 + pad opening + rdl landing."""
    c = gdstk.Cell(name)
    rect(c, 0, 0, 80, 80, "met3")
    rect(c, 10, 10, 60, 60, "pad")
    rect(c, 0, 0, 80, 80, "rdl")
    return c


def add_label(cell, text, x, y):
    l, dt = LAYERS["label"]
    cell.add(gdstk.Label(text, (x, y), layer=l, texttype=dt))


def build_bandgap_block():
    """
    BANDGAP_REF subcell: two matched PNP BJTs Q1, Q2 side by side,
    a current mirror (M1, M2), and a reference resistor chain.

    Size ~200 x 120 um.
    """
    bjt = bjt_cell("BJT_PNP")
    mos = mosfet_cell("MOS_PMOS")
    res = resistor_cell("RES_POLY")

    c = gdstk.Cell("BANDGAP_REF")

    # Matched PNP pair — Q1 and Q2 — placed at y=20, 40um apart
    # This is the sensitive matched region the client's PPT highlights.
    c.add(gdstk.Reference(bjt, origin=(20, 20)))      # Q1
    add_label(c, "Q1", 35, 52)
    c.add(gdstk.Reference(bjt, origin=(70, 20)))      # Q2 (40um to right)
    add_label(c, "Q2", 85, 52)

    # Current mirror — M1 / M2 above the BJTs
    c.add(gdstk.Reference(mos, origin=(120, 40)))     # M1
    add_label(c, "M1", 128, 50)
    c.add(gdstk.Reference(mos, origin=(150, 40)))     # M2
    add_label(c, "M2", 158, 50)

    # Reference resistors (string of 3)
    for i in range(3):
        c.add(gdstk.Reference(res, origin=(20 + i * 55, 80)))
    add_label(c, "R_ref", 50, 90)

    # Local met2 rail tying Q1/Q2 emitters
    rect(c, 20, 53, 80, 3, "met2")

    # Block boundary marker using met3 outline (visual)
    # (not strictly required; helps rendering)
    outline = gdstk.rectangle((0, 0), (200, 120),
                              layer=LAYERS["met3"][0], datatype=1)  # dt=1 = outline style
    c.add(outline)

    # Block-level label
    add_label(c, "BANDGAP_REF", 80, 110)
    return c


def build_other_block():
    """A second, less-sensitive block (e.g. digital/IO) to add realism."""
    mos = mosfet_cell("MOS_PMOS")
    c = gdstk.Cell("DIGITAL_BLOCK")
    # Row of inverters
    for i in range(6):
        c.add(gdstk.Reference(mos, origin=(10 + i * 25, 10)))
    # Some met1/met2 routing
    rect(c, 10, 30, 150, 3, "met1")
    rect(c, 10, 35, 150, 3, "met2")
    add_label(c, "DIGITAL_BLOCK", 60, 45)
    return c


def build_chip():
    """
    Top-level chip: 600 x 400 um with:
      - BANDGAP_REF block at (50, 80)
      - DIGITAL_BLOCK at (300, 80)
      - Pads around periphery
      - RDL strip crossing over the BANDGAP_REF matched pair <-- planted issue
    """
    lib = gdstk.Library(unit=1e-6, precision=1e-9)

    bg = build_bandgap_block()
    dig = build_other_block()
    pad = pad_cell("PAD")

    # Register subcells
    for c in [bg, dig, pad]:
        lib.add(c)
    # Register all referenced leaf cells
    for leaf_name in ["BJT_PNP", "MOS_PMOS", "RES_POLY"]:
        for c in lib.cells:
            if c.name == leaf_name:
                break
        else:
            # gdstk References already carry refs to the leaf cells;
            # they need to be in the library so writer serialises them
            pass
    # Walk references to collect all cells
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

    top = gdstk.Cell("analog_chip_top")
    lib.add(top)

    # Blocks
    top.add(gdstk.Reference(bg, origin=(50, 80)))
    top.add(gdstk.Reference(dig, origin=(300, 80)))

    # Pads around periphery (8 pads along top edge, 80x80 each, 20 apart)
    pad_y = 300
    pad_names = ["VIN", "VOUT", "VREF", "GND", "EN", "SDA", "SCL", "VDDA"]
    for i, pname in enumerate(pad_names):
        x = 30 + i * 70
        top.add(gdstk.Reference(pad, origin=(x, pad_y)))
        add_label(top, pname, x + 20, pad_y + 35)

    # Die outline
    rect(top, 0, 0, 600, 400, "met3", datatype=1)

    # === THE PLANTED ISSUE ===
    # RDL strip crossing horizontally over the BANDGAP_REF matched pair.
    # BANDGAP_REF is at (50, 80), Q1 at local (20,20) -> global (70, 100),
    # Q2 at local (70,20) -> global (120, 100). Each Q is 30x30.
    # So we drop an RDL strip at y = 95..115 covering x = 30..260 — clearly
    # on top of both Q1 and Q2.
    rect(top, 30, 95, 230, 20, "rdl")
    add_label(top, "RDL_VIN_trace", 145, 105)

    # Also add an RDL landing pad on VIN pad at top (benign — this is ok)
    rect(top, 30, pad_y, 80, 80, "rdl")

    # Top-level cell label
    add_label(top, "analog_chip_top", 300, 380)

    return lib


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "demo_chip.gds")
    lib = build_chip()
    lib.write_gds(out)
    print(f"Wrote {out}")
    # Quick stats
    for c in lib.cells:
        pg = len(c.polygons) if hasattr(c, "polygons") else 0
        rf = len(c.references) if hasattr(c, "references") else 0
        lb = len(c.labels) if hasattr(c, "labels") else 0
        print(f"  cell {c.name}: polys={pg} refs={rf} labels={lb}")
