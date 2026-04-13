# Example Usage

## Full workflow for a new GDS file

### 1. Explore the file

```bash
$ python explore_gds.py ~/tapeouts/MPQ8897_R4.gds

Loading ~/tapeouts/MPQ8897_R4.gds ...
Loaded library with 247 cells.

=== FILE SUMMARY ===
  Total cells:      247
  Top-level cells:  1
    - MPQ8897_TOP

=== INSPECTING CELL: MPQ8897_TOP ===
  Bounding box: (-50.000, -50.000) to (2200.000, 2200.000)
  Width x Height: 2250.000 x 2250.000
  Polygons:    1523
  Paths:       0
  References:  89
  Labels:      34

=== LAYER SUMMARY (flattened) ===
   Layer  Datatype  Polygons     Total Area
   -----  --------  ---------  ----------
       1         0     12034     500000.00
      10         0      8921     320000.00
      30         0      4521     125000.50
      31         0      3892      98000.25
      40         0      6234     450000.00
      41         0      5100     380000.00
      70         0        24      18000.00
      71         0        12       9600.00

  Unique layer/datatype pairs: 8

=== CELL HIERARCHY (root: MPQ8897_TOP) ===
  ├── PAD_CELL x24
  ├── DCDC_CORE
  │   ├── POWER_STAGE
  │   ├── CONTROL_LOGIC
  │   └── COMP_BLOCK x4
  ├── ESD_CELL x24
  └── DECAP_ARRAY x8

Done.
```

### 2. Check hierarchy only

```bash
$ python explore_gds.py ~/tapeouts/MPQ8897_R4.gds --hierarchy
```

### 3. Export layer visuals

```bash
$ python explore_gds.py ~/tapeouts/MPQ8897_R4.gds --export-layers ./layer_svgs/
```

This creates one SVG per layer in `./layer_svgs/` — useful for quickly seeing what each layer looks like.

### 4. Create layer config

After looking at the layer summary and checking process docs, create `layer_config.json`:

```json
{
  "pad_layers": [[70, 0]],
  "rdl_layers": [[71, 0]],
  "check_layers": [
    {"layer": [30, 0], "name": "poly"},
    {"layer": [31, 0], "name": "active"},
    {"layer": [40, 0], "name": "metal1"},
    {"layer": [41, 0], "name": "metal2"}
  ],
  "min_overlap_area": 0.1
}
```

### 5. Run overlap detection

```bash
$ python find_overlaps.py ~/tapeouts/MPQ8897_R4.gds --config layer_config.json --output results.json

Loading ~/tapeouts/MPQ8897_R4.gds ...
  Loaded in 2.3s — 247 cells
Flattening cell hierarchy ...
  Flattened in 1.1s
Extracting polygons by layer ...
  45702 polygons across 8 layer/datatype pairs
Detecting overlaps ...
  Processing region 24/24 ...
  Found 7 overlaps in 0.4s
Results written to results.json
```

### 6. Generate markdown report

```bash
$ python report_generator.py results.json --output report.md
Markdown report written to report.md
```

### 7. Generate LLM prompt

```bash
$ python report_generator.py results.json --llm-prompt > prompt.txt
```

Then paste `prompt.txt` into Claude or another LLM for risk assessment. The prompt includes all findings plus context about known IC packaging failure modes.

## Example layer_config.json for different processes

Each foundry/process has different layer numbers. Here are templates:

### Generic (fill in your numbers)

```json
{
  "pad_layers": [[0, 0]],
  "rdl_layers": [],
  "check_layers": [
    {"layer": [0, 0], "name": "poly"},
    {"layer": [0, 0], "name": "active"},
    {"layer": [0, 0], "name": "metal1"}
  ],
  "min_overlap_area": 0.1
}
```

### Tips for identifying layers

1. **Run `explore_gds.py --layers`** — pad layers typically have few polygons (one per pad) with large individual areas
2. **Open in KLayout** — toggle layers on/off to visually identify them
3. **Check process docs** — your PDK documentation should have a layer map
4. **Count polygons** — poly layers have many small polygons, metal layers fewer but larger ones
