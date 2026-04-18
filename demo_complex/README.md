# demo_complex — Claude vs Qwen on a complex synthetic chip

Five-block synthetic analog IC (bandgap, LDO, comparator, LC oscillator,
ESD IO) with FOUR distinct planted issues plus ONE intentional false
positive. The same detector + same renderer is driven by two different
backends so the comparison is purely about review *quality*.

## Run on the work machine (Qwen, live)

```
python run_complex_demo.py --backend qwen
```

Hits `http://localhost:8000/v1/chat/completions` by default; auto-discovers
the model id from `/v1/models`. Output: `demo_complex/complex_report_qwen.png`.

## Reference (Claude, cached)

`claude_complex_report.png` at the repo root is the Claude-authored
reference, generated from `demo_complex/claude_responses.json`.

To regenerate:

```
python run_complex_demo.py --backend claude --copy-to ./claude_complex_report.png
```

## What's planted

| # | Block       | Mechanism                                              | Detector rule                  | Reviewer should... |
|---|-------------|--------------------------------------------------------|--------------------------------|--------------------|
| 1 | BANDGAP_REF | RDL strip over Q1/Q2 matched BJT pair                  | rdl_over_bjt_pair (critical)   | Explain thermal + piezo + capacitive coupling |
| 2 | LDO_REG     | Thick top-metal current rail over R_fb divider         | thick_metal_over_poly (critical)| Explain joule heating asymmetry on matched divider |
| 3 | COMP_INPUT  | Dummy poly fill on Mi_p only, not Mi_n                 | poly_fill_over_input_diff (high)| Explain input-pair asymmetry, fix at fill tool |
| 4 | OSC_LC      | CLK_DIG met2 line crossing high-Z tank node, no shield | digital_signal_near_hi_z (high)| Explain capacitive injection, spur/jitter |
| 5 | ESD_IO      | RDL from PAD_VDD down to ESD_CLAMP_NMOS                | rdl_over_diff (high)            | Recognize as INTENDED — flag as false positive |

The detector also produces two duplicate findings (`rdl_over_diff` inside
BANDGAP_REF — same RDL as #1; `rdl_over_poly` inside ESD_IO — same pad
strap as #5). A good reviewer clusters those into the parent finding
instead of writing two more cards.

## What separates a good review from a generic one

- Cluster duplicates rather than re-writing the same analysis twice.
- Recognize the planted false positive instead of escalating it.
- Pick the *right* fix per mechanism — shielding doesn't fix a thermal
  gradient; a fill-tool problem isn't a routing problem.
- Stay terse — quote netlist evidence, don't pad.
