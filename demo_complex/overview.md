# GDS overview — complex_chip.gds

_Generated in 0.0s. Answers David's two questions, decomposed into six sections._

## 1. What's in this GDS?

| Field | Value |
|---|---|
| Top cell | `analog_ip_top` |
| Cells in library | 14 |
| Die bounding box | (0.0, 0.0) → (1200.0, 800.0) |
| Die size | 1200.0 × 800.0 µm (area 960000 µm²) |
| Hierarchy max depth | 3 levels |
| Unique layer/datatype pairs | 14 |
| Total polygons (all layers) | 141 |

## 2. What layers exist?

Every layer/datatype pair in the file, sorted by category (pad candidates first). Categories are heuristic from geometry — confirm with the designer if ambiguous.

| Layer | Category | Conf. | Polys | Die % | Mean area (µm²) | % rect | % near edge | Reasoning |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/0 | passivation_or_pad | 50% | 10 | 1.2% | 1149.6 | 100 | 70 | medium rects, 70% near edge |
| 2/0 | passivation_or_pad | 50% | 3 | 0.5% | 1666.7 | 100 | 33 | medium rects, 33% near edge |
| 33/0 | passivation_or_pad | 50% | 5 | 3.3% | 6400.0 | 100 | 100 | medium rects, 100% near edge |
| 50/0 | passivation_or_pad | 50% | 7 | 4.2% | 5771.4 | 100 | 100 | medium rects, 100% near edge |
| 60/0 | passivation_or_pad | 50% | 5 | 1.9% | 3600.0 | 100 | 100 | medium rects, 100% near edge |
| 90/0 | passivation_or_pad | 50% | 1 | 0.4% | 3840.0 | 100 | 100 | medium rects, 100% near edge |
| 33/1 | top_metal_or_rdl | 65% | 6 | 111.8% | 178900.0 | 100 | 100 | 6 polys, 112% die coverage |
| 6/0 | marker_or_text | 40% | 3 | 0.0% | 36.0 | 100 | 100 | 3 polys, negligible area |
| 22/0 | marker_or_text | 40% | 1 | 0.1% | 500.0 | 100 | 100 | 1 polys, negligible area |
| 5/0 | unclassified | 20% | 30 | 0.6% | 182.7 | 100 | 47 | 30 polys, mean area 183, 0.6% die |
| 10/0 | unclassified | 20% | 43 | 0.0% | 7.3 | 100 | 33 | 43 polys, mean area 7, 0.0% die |
| 11/0 | unclassified | 20% | 24 | 1.1% | 457.0 | 100 | 50 | 24 polys, mean area 457, 1.1% die |
| 40/0 | unclassified | 20% | 1 | 0.7% | 6600.0 | 100 | 0 | 1 polys, mean area 6600, 0.7% die |
| 82/5 | unclassified | 20% | 2 | 0.2% | 900.0 | 100 | 50 | 2 polys, mean area 900, 0.2% die |

## 3. Which layer is the pad / bump?

**Chosen pad layer:** `50/0` — user-specified

**All pad-like candidates (in order of confidence):**

| Layer | Category | Conf. | Polys | Mean area (µm²) | % near edge |
|---|---|---:|---:|---:|---:|
| 1/0 | passivation_or_pad | 50% | 10 | 1149.6 | 70% |
| 2/0 | passivation_or_pad | 50% | 3 | 1666.7 | 33% |
| 33/0 | passivation_or_pad | 50% | 5 | 6400.0 | 100% |
| 50/0 | passivation_or_pad | 50% | 7 | 5771.4 | 100% |
| 60/0 | passivation_or_pad | 50% | 5 | 3600.0 | 100% |
| 90/0 | passivation_or_pad | 50% | 1 | 3840.0 | 100% |

## 4. What device layers exist?

**Chosen device layers:** `1/0`, `5/0`, `82/5` — user-specified

_No device-like candidates auto-detected. Pass explicit layers via `--device-layers a/b,c/d`._

## 5. Where are the pads?

**6 pad area(s) found on layer 50/0.**
2 have device overlap or a nearby device (buffer 5.0 µm); 4 are clean.


| # | Center (µm) | Size (µm) | Container cell | Under? | Near? |
|---:|---|---|---|---:|---:|
| 1 | (100.9, 684.3) | 80.0 × 190.0 | `analog_ip_top` | 2 | 0 |
| 2 | (175.0, 145.0) | 210.0 × 20.0 | `analog_ip_top` | 2 | 0 |
| 3 | (340.0, 720.0) | 80.0 × 80.0 | `analog_ip_top/PAD` | 0 | 0 |
| 4 | (580.0, 720.0) | 80.0 × 80.0 | `analog_ip_top/PAD` | 0 | 0 |
| 5 | (820.0, 720.0) | 80.0 × 80.0 | `analog_ip_top/PAD` | 0 | 0 |
| 6 | (1060.0, 720.0) | 80.0 × 80.0 | `analog_ip_top/PAD` | 0 | 0 |

## 6. What's under / near each pad?

### PAD #1 — at (100.9, 684.3), 80.0 × 190.0 µm
Container cell: `analog_ip_top`

**Directly under:**

| Cell (owner path) | Layer | Overlap area (µm²) |
|---|---|---:|
| `analog_ip_top/ESD_IO/ESD_CLAMP_NMOS` | 1/0 (passivation_or_pad) | 1200.0 |
| `analog_ip_top/ESD_IO/ESD_CLAMP_NMOS` | 5/0 (unclassified) | 372.0 |

**Within 5.0 µm:**

_Nothing within buffer._

---

### PAD #2 — at (175.0, 145.0), 210.0 × 20.0 µm
Container cell: `analog_ip_top`

**Directly under:**

| Cell (owner path) | Layer | Overlap area (µm²) |
|---|---|---:|
| `analog_ip_top/BANDGAP_REF/BJT_PNP` | 82/5 (unclassified) | 900.0 |
| `analog_ip_top/BANDGAP_REF/BJT_PNP` | 1/0 (passivation_or_pad) | 612.0 |

**Within 5.0 µm:**

_Nothing within buffer._

---

### PAD #3 — at (340.0, 720.0), 80.0 × 80.0 µm
Container cell: `analog_ip_top/PAD`

**Directly under:**

_Nothing under._

**Within 5.0 µm:**

_Nothing within buffer._

---

### PAD #4 — at (580.0, 720.0), 80.0 × 80.0 µm
Container cell: `analog_ip_top/PAD`

**Directly under:**

_Nothing under._

**Within 5.0 µm:**

_Nothing within buffer._

---

### PAD #5 — at (820.0, 720.0), 80.0 × 80.0 µm
Container cell: `analog_ip_top/PAD`

**Directly under:**

_Nothing under._

**Within 5.0 µm:**

_Nothing within buffer._

---

### PAD #6 — at (1060.0, 720.0), 80.0 × 80.0 µm
Container cell: `analog_ip_top/PAD`

**Directly under:**

_Nothing under._

**Within 5.0 µm:**

_Nothing within buffer._

---
