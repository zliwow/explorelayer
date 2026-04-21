#!/usr/bin/env python3
"""
md_to_pdf.py — turn a markdown file into a PDF using the pip `markdown-pdf`
library. Pure Python, no Chromium / LaTeX needed on the host.

Usage:
    python utils/md_to_pdf.py davids_overview.md
    python utils/md_to_pdf.py davids_overview.md out.pdf

Dependency:
    pip install markdown-pdf
"""

import os
import sys

try:
    from markdown_pdf import MarkdownPdf, Section
except ImportError:
    sys.exit("Missing dep — run: pip install markdown-pdf")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python utils/md_to_pdf.py input.md [output.pdf]")
    md_path = sys.argv[1]
    if not os.path.exists(md_path):
        sys.exit(f"Not found: {md_path}")

    out_path = (sys.argv[2] if len(sys.argv) > 2
                else os.path.splitext(md_path)[0] + ".pdf")

    pdf = MarkdownPdf(toc_level=2)
    with open(md_path, "r", encoding="utf-8") as f:
        pdf.add_section(Section(f.read()))
    pdf.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
