#!/usr/bin/env python3
"""
convert_pdfs.py — Batch PDF → Markdown converter for the Linguistics CLI pipeline.

Uses the `marker` library (https://github.com/VikParuchuri/marker) to convert
research paper PDFs into clean Markdown files that ManuscriptLoader can ingest.

The model bundle is loaded **once** and reused across all files in the batch,
which is critical for performance — loading models per-file takes minutes.

Usage:
    python convert_pdfs.py <input_dir> <output_dir> [--workers N]

Arguments:
    input_dir   Directory containing .pdf files to convert
    output_dir  Directory where .md files will be written (created if absent)
    --workers   Number of parallel CPU workers (default: 4)

Requirements:
    pip install marker-pdf

Example:
    python convert_pdfs.py ~/papers/raw ~/papers/markdown
    python convert_pdfs.py ~/papers/raw ~/papers/markdown --workers 8
"""

import argparse
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a folder of PDFs to Markdown using marker."
    )
    parser.add_argument("input_dir",  help="Directory containing PDF files")
    parser.add_argument("output_dir", help="Directory for output Markdown files")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Number of parallel CPU workers (default: 4)")
    args = parser.parse_args()

    input_dir  = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    pdfs = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {input_dir}")
    print("Loading marker model bundle (this may take 30–60 s on first run)…")

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_bundle
    except ImportError:
        print(
            "Error: marker-pdf is not installed.\n"
            "Install it with:  pip install marker-pdf",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load models once — expensive; reused across all files.
    models = create_model_bundle()
    converter = PdfConverter(artifact_dict=models)
    print("Model bundle ready.\n")

    ok = 0
    failed = []

    for fname in pdfs:
        src_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".md"
        out_path = os.path.join(output_dir, out_name)

        start = time.time()
        try:
            result = converter(src_path)
            # marker returns (markdown_text, metadata, images) or a single string
            # depending on the version — handle both shapes.
            if isinstance(result, tuple):
                markdown = result[0]
            else:
                markdown = str(result)

            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(markdown)

            elapsed = time.time() - start
            print(f"  ✓ {fname}  →  {out_name}  ({elapsed:.1f}s)")
            ok += 1

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            print(f"  ✗ {fname}  FAILED after {elapsed:.1f}s: {exc}", file=sys.stderr)
            failed.append(fname)

    print(f"\nDone — {ok} converted, {len(failed)} failed.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
