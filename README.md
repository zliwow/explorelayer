# gds-layout-explorer

GDS-only tools for reviewing analog/power-IC layouts: find pads, see what
sits under them, and flag hierarchy-attributed overlaps between top-metal
routing and sensitive devices.

## Top-level scripts

| Script                  | Purpose                                                      |
|-------------------------|--------------------------------------------------------------|
| `pad_scan.py`           | Identify pad/top-layer areas and list devices under/near each pad. Single GDS in, single markdown out. |
| `find_overlaps_v2.py`   | Hierarchy-aware aggressor→victim overlap detector. Outputs `findings_v2.json`. |
| `report_v2.py`          | Turn `findings_v2.json` into `report.md` + per-finding LLM `analysis.md`. |
| `run_complex_demo.py`   | End-to-end demo: generate complex_chip, detect, render PNG.  |
| `run_qwen_demo.py`      | End-to-end demo: synthetic chip, detect, render PNG.         |

## Directory layout

```
.
├── pad_scan.py              pad + devices-nearby scanner (GDS only)
├── find_overlaps_v2.py      v2 detector
├── report_v2.py             v2 reporter
├── run_complex_demo.py      complex demo orchestrator
├── run_qwen_demo.py         synthetic demo orchestrator
├── demo_complex/            five-block synthetic chip + planted faults
├── demo_synthetic/          simpler single-block synthetic chip
├── utils/                   exploration helpers (dump labels, list subckts, etc.)
├── legacy_v1/               pre-v2 pipeline; kept for the 39/39-critical baseline
├── examples/                usage notes
└── requirements.txt
```

## Quick start

```bash
# Pad-area / devices-under-pad scan (GDS only, no netlist needed)
python pad_scan.py demo_complex/complex_chip.gds --pad-layer 50/0

# Full v2 detector + markdown report
python find_overlaps_v2.py demo_complex/complex_chip.gds \
    --config demo_complex/layer_config.json \
    --output findings_v2.json
python report_v2.py findings_v2.json           # writes report.md + analysis.md
python report_v2.py findings_v2.json --no-llm  # skip LLM, write report.md only

# Complex chip demo (generate → detect → render PNG)
python run_complex_demo.py --backend claude
python run_complex_demo.py --backend qwen
```

## Requirements

```
pip install -r requirements.txt
```

`report_v2.py` and the demo renderers expect an OpenAI-compatible LLM
endpoint at `http://localhost:8000/v1` (defaults to auto-discovering the
model id from `/v1/models`). Pass `--no-llm` or `--backend claude` to
skip live inference.
