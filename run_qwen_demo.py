#!/usr/bin/env python3
"""
run_qwen_demo.py — one-shot Qwen reproduction of the Claude demo.

Runs the whole synthetic-chip pipeline end-to-end:

  1. demo_synthetic/make_demo_gds.py   — generate the planted-issue GDS
  2. demo_synthetic/find_issues.py     — detector (RDL over sensitive cells)
  3. demo_synthetic/render_report.py   — render PNG with Qwen-authored prose

Output: demo_synthetic/demo_report.png. Compare side-by-side against
reference_demo_report.png (Claude-authored on the same chip) to see how
a local Qwen server differs from Claude on a ground-truth layout.

Usage:
  python run_qwen_demo.py
  python run_qwen_demo.py --llm-url http://localhost:8000/v1/chat/completions
  python run_qwen_demo.py --skip-gen           # reuse existing demo_chip.gds
  python run_qwen_demo.py --output mine.png
"""

import argparse
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo_synthetic")


def run(cmd, cwd):
    print(f"\n$ {' '.join(cmd)}   (cwd={os.path.relpath(cwd, HERE) or '.'})")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        sys.exit(f"step failed with exit code {proc.returncode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip regenerating demo_chip.gds (reuse existing).")
    p.add_argument("--skip-detect", action="store_true",
                   help="Skip rerunning the detector (reuse existing findings.json).")
    p.add_argument("--llm-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default=None,
                   help="Default: auto-discover from /v1/models.")
    p.add_argument("--output", default="demo_report.png",
                   help="PNG filename written inside demo_synthetic/.")
    p.add_argument("--copy-to", default=None,
                   help="Optional: also copy the final PNG to this path "
                        "(e.g. ./qwen_demo_report.png at the repo root).")
    args = p.parse_args()

    if not os.path.isdir(DEMO_DIR):
        sys.exit(f"Missing {DEMO_DIR}. Did you run `git pull`?")

    py = sys.executable or "python3"
    gds_path = os.path.join(DEMO_DIR, "demo_chip.gds")
    findings_path = os.path.join(DEMO_DIR, "findings.json")
    out_path = os.path.join(DEMO_DIR, args.output)

    # Step 1 — generate the synthetic chip
    if args.skip_gen and os.path.exists(gds_path):
        print(f"[skip] demo_chip.gds already exists at {gds_path}")
    else:
        run([py, "make_demo_gds.py"], cwd=DEMO_DIR)

    # Step 2 — detector
    if args.skip_detect and os.path.exists(findings_path):
        print(f"[skip] findings.json already exists at {findings_path}")
    else:
        run([py, "find_issues.py", "demo_chip.gds",
             "--config", "layer_config.json",
             "--output", "findings.json"], cwd=DEMO_DIR)

    # Step 3 — render with Qwen
    render_cmd = [py, "render_report.py",
                  "--findings", "findings.json",
                  "--gds", "demo_chip.gds",
                  "--netlist", "demo_chip.cdl",
                  "--output", args.output,
                  "--llm-url", args.llm_url]
    if args.llm_model:
        render_cmd += ["--llm-model", args.llm_model]
    run(render_cmd, cwd=DEMO_DIR)

    print(f"\nDone. Output: {out_path}")

    if args.copy_to:
        dest = args.copy_to
        if not os.path.isabs(dest):
            dest = os.path.join(HERE, dest)
        shutil.copy2(out_path, dest)
        print(f"Also copied to: {dest}")

    ref = os.path.join(HERE, "reference_demo_report.png")
    if os.path.exists(ref):
        print(f"\nCompare to Claude reference: {os.path.relpath(ref, HERE)}")


if __name__ == "__main__":
    main()
