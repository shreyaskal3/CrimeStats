#!/usr/bin/env python3
"""Run layout -> extract tables -> merge, then keep only page 1 tables."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run layout analysis and extract tables for all PDFs, then merge page 1 tables."
    )
    parser.add_argument("--pdf-dir", default="crimepdf1")
    parser.add_argument("--layout-dir", default="layout_outputs")
    parser.add_argument("--extracted-root", default="extracted_tables")
    parser.add_argument("--converted-dir", default="converted_page1")
    parser.add_argument("--input-csv-name", default="table_1_page_1.csv")
    parser.add_argument("--skip-layout", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    layout_dir = Path(args.layout_dir)
    extracted_root = Path(args.extracted_root)
    converted_dir = Path(args.converted_dir)
    input_csv_name = args.input_csv_name

    layout_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return 1

    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        layout_json = layout_dir / f"{stem}.json"
        out_dir = extracted_root / stem

        if layout_json.exists():
            print(f"Skipping layout (exists): {layout_json}")
        elif not args.skip_layout:
            run([sys.executable, "run_layout_analysis.py", str(pdf_path), "--out", str(layout_json)])
        else:
            print(f"Missing layout JSON (skip-layout set): {layout_json}")
            return 1

        extracted_json = out_dir / "extracted_tables.json"
        if extracted_json.exists():
            print(f"Skipping extract (exists): {extracted_json}")
        elif not args.skip_extract:
            run([sys.executable, "extract_tables.py", str(layout_json), "--outdir", str(out_dir)])
        else:
            print(f"Missing extracted_tables.json (skip-extract set): {out_dir}")
            return 1

        input_csv = out_dir / input_csv_name
        if not input_csv.exists():
            print(f"Missing input CSV: {input_csv}")
            continue

        output_csv = converted_dir / f"crime_monthly_stats_{stem}.csv"
        env = os.environ.copy()
        env["INPUT_CSV"] = str(input_csv)
        env["OUTPUT_CSV"] = str(output_csv)
        print("+", f"INPUT_CSV={input_csv} OUTPUT_CSV={output_csv} python3 convertcsv.py")
        try:
            subprocess.run(
                [sys.executable, "convertcsv.py"],
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            print(f"convertcsv.py failed for {input_csv} (exit {exc.returncode}). Skipping.")
            continue

    print(f"Done. Converted outputs: {converted_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
