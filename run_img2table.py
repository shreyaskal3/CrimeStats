from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse
from pathlib import Path

from img2table.document import PDF
from img2table.ocr import TesseractOCR


def parse_pages(pages_str: str | None) -> list[int] | None:
    if not pages_str:
        return None
    pages = []
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return pages or None


def build_ocr(args: argparse.Namespace):
    if args.ocr == "tesseract":
        return TesseractOCR(n_threads=args.threads, lang=args.lang)
    if args.ocr == "azure":
        try:
            from img2table.ocr import AzureOCR
        except Exception as exc:
            print(f"Azure OCR not available: {exc}", file=sys.stderr)
            print("Install with: python3 -m pip install 'img2table[azure]'", file=sys.stderr)
            raise SystemExit(2)

        endpoint = args.azure_endpoint or os.environ.get("COMPUTER_VISION_ENDPOINT")
        key = args.azure_key or os.environ.get("COMPUTER_VISION_SUBSCRIPTION_KEY")
        if not endpoint or not key:
            try:
                import config
            except Exception:
                config = None
            if config is not None:
                layout_url = getattr(config, "layout_url", None)
                layout_key = getattr(config, "layout_key", None)
                if layout_url and layout_key:
                    parsed = urlparse(layout_url)
                    endpoint = endpoint or f"{parsed.scheme}://{parsed.netloc}/"
                    key = key or layout_key
        if not endpoint or not key:
            print("Missing Azure credentials.", file=sys.stderr)
            print("Provide --azure-endpoint/--azure-key or set", file=sys.stderr)
            print("COMPUTER_VISION_ENDPOINT and COMPUTER_VISION_SUBSCRIPTION_KEY.", file=sys.stderr)
            print("You can also define layout_url/layout_key in config.py.", file=sys.stderr)
            raise SystemExit(2)
        return AzureOCR(endpoint=endpoint, subscription_key=key)

    print(f"Unknown OCR backend: {args.ocr}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract tables from a PDF using img2table.")
    parser.add_argument("--pdf", default="crime_pdfs/1.pdf", help="Path to PDF")
    parser.add_argument("--out", default="extracted_tables/img2table_1.xlsx", help="Output XLSX path")
    parser.add_argument("--pages", default=None, help="Pages to process (e.g. 0,2,4-6)")
    parser.add_argument("--threads", type=int, default=1, help="Tesseract threads")
    parser.add_argument("--lang", default="eng", help="Tesseract language")
    parser.add_argument("--ocr", default="tesseract", choices=["tesseract", "azure"], help="OCR backend")
    parser.add_argument("--azure-endpoint", default=None, help="Azure Computer Vision endpoint")
    parser.add_argument("--azure-key", default=None, help="Azure Computer Vision subscription key")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless table detection")
    parser.add_argument("--implicit-rows", action="store_true", help="Allow implicit rows")
    parser.add_argument("--min-confidence", type=int, default=50, help="Minimum OCR confidence (0-100)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = parse_pages(args.pages)
    ocr = build_ocr(args)

    pdf = PDF(src=str(pdf_path), pages=pages)
    extracted = pdf.extract_tables(
        ocr=ocr,
        borderless_tables=args.borderless,
        implicit_rows=args.implicit_rows,
        min_confidence=args.min_confidence,
    )
    pdf.to_xlsx(
        dest=str(out_path),
        ocr=ocr,
        borderless_tables=args.borderless,
        implicit_rows=args.implicit_rows,
        min_confidence=args.min_confidence,
    )

    num_pages = len(extracted)
    num_tables = sum(len(v) for v in extracted.values())
    print(f"Pages with tables: {num_pages}")
    print(f"Total tables: {num_tables}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
