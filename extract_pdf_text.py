#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def extract_pymupdf(pdf_path: Path) -> tuple[bool, str]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        return False, f"ERROR: PyMuPDF unavailable: {exc}"
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        return True, text
    except Exception as exc:
        return False, f"ERROR: PyMuPDF failed: {exc}"


def extract_pdfminer(pdf_path: Path) -> tuple[bool, str]:
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:
        return False, f"ERROR: pdfminer unavailable: {exc}"
    try:
        text = extract_text(str(pdf_path))
        return True, text
    except Exception as exc:
        return False, f"ERROR: pdfminer failed: {exc}"


def extract_pdftotext(pdf_path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return False, f"ERROR: pdftotext not found: {exc}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        return False, f"ERROR: pdftotext failed: {stderr or exc}"
    return True, result.stdout


def write_output(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF with PyMuPDF, pdfminer, and pdftotext.",
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="crime_pdfs/1.pdf",
        help="Path to the PDF (default: crime_pdfs/1.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        default="parsed_outputs",
        help="Directory for output files (default: parsed_outputs)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    results: list[tuple[str, bool, str]] = []

    ok, text = extract_pymupdf(pdf_path)
    results.append(("PyMuPDF", ok, text))
    if ok:
        write_output(output_dir / f"{stem}_pymupdf.txt", text)

    ok, text = extract_pdfminer(pdf_path)
    results.append(("pdfminer", ok, text))
    if ok:
        write_output(output_dir / f"{stem}_pdfminer.txt", text)

    ok, text = extract_pdftotext(pdf_path)
    results.append(("pdftotext -layout", ok, text))
    if ok:
        write_output(output_dir / f"{stem}_pdftotext.txt", text)

    combined_parts: list[str] = []
    for label, _, text in results:
        combined_parts.append(f"===== {label} =====\n")
        combined_parts.append(text)
        if not text.endswith("\n"):
            combined_parts.append("\n")
        combined_parts.append("\n")

    combined_path = output_dir / f"{stem}_combined.txt"
    write_output(combined_path, "".join(combined_parts))

    for label, ok, _ in results:
        status = "ok" if ok else "failed"
        print(f"{label}: {status}")
    print(f"Wrote combined: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
