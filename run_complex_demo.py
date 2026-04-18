#!/usr/bin/env python3
"""
run_complex_demo.py — orchestrator for the Claude-vs-Qwen comparison.

  python run_complex_demo.py --backend claude    # uses cached Claude prose
  python run_complex_demo.py --backend qwen      # live Qwen at localhost:8000

Both backends drive the SAME pipeline:
  1. demo_complex/make_complex_chip.py   (skip with --skip-gen)
  2. demo_complex/find_issues.py         (skip with --skip-detect)
  3. demo_complex/render_complex.py --backend <backend>

The two output PNGs sit side-by-side so the difference is purely the
backend's review quality, not the chrome.
"""

import argparse
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_complex")


def run(cmd, cwd):
    print(f"\n$ {' '.join(cmd)}   (cwd={os.path.relpath(cwd, HERE) or '.'})")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        sys.exit(f"step failed with exit code {proc.returncode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["claude", "qwen"], required=True)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--llm-url",
                   default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--output", default=None,
                   help="Output PNG name inside demo_complex/. "
                        "Default: complex_report_<backend>.png")
    p.add_argument("--copy-to", default=None,
                   help="Optional: also copy the final PNG to this path "
                        "(e.g. ./qwen_complex_report.png at repo root).")
    args = p.parse_args()

    if not os.path.isdir(DEMO):
        sys.exit(f"Missing {DEMO}. Did you pull demo_complex/?")

    py = sys.executable or "python3"
    gds_path = os.path.join(DEMO, "complex_chip.gds")
    findings_path = os.path.join(DEMO, "findings.json")
    out_name = args.output or f"complex_report_{args.backend}.png"
    out_path = os.path.join(DEMO, out_name)

    # 1. Generate the chip
    if args.skip_gen and os.path.exists(gds_path):
        print(f"[skip] {gds_path} exists")
    else:
        run([py, "make_complex_chip.py"], cwd=DEMO)

    # 2. Run detector
    if args.skip_detect and os.path.exists(findings_path):
        print(f"[skip] {findings_path} exists")
    else:
        run([py, "find_issues.py", "complex_chip.gds",
             "--config", "layer_config.json",
             "--output", "findings.json"], cwd=DEMO)

    # 3. Render with chosen backend
    cmd = [py, "render_complex.py",
           "--gds", "complex_chip.gds",
           "--findings", "findings.json",
           "--netlist", "complex_chip.cdl",
           "--backend", args.backend,
           "--output", out_name]
    if args.backend == "qwen":
        cmd += ["--llm-url", args.llm_url]
        if args.llm_model:
            cmd += ["--llm-model", args.llm_model]
    run(cmd, cwd=DEMO)

    print(f"\nDone. Output: {out_path}")

    if args.copy_to:
        dest = args.copy_to
        if not os.path.isabs(dest):
            dest = os.path.join(HERE, dest)
        shutil.copy2(out_path, dest)
        print(f"Also copied to: {dest}")


if __name__ == "__main__":
    main()
