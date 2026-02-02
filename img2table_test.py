from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from img2table.document import PDF
from img2table.ocr import AzureOCR

import config


def get_page_count(pdf_path: Path) -> int:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return 0
    doc = pdfium.PdfDocument(str(pdf_path))
    return len(doc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract tables from all pages using Azure OCR.")
    parser.add_argument("--pdf", default="crime_pdfs/1.pdf", help="Path to PDF")
    parser.add_argument("--out-dir", default="extracted_tables", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Pages per batch")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless table detection")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(config.layout_url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}/"
    key = config.layout_key

    ocr = AzureOCR(endpoint=endpoint, subscription_key=key)

    total_pages = get_page_count(pdf_path)
    if total_pages <= 0:
        total_pages = 1

    batch_size = max(1, args.batch_size)
    for start in range(0, total_pages, batch_size):
        end = min(total_pages, start + batch_size)
        pages = list(range(start, end))
        pdf = PDF(src=str(pdf_path), pages=pages)

        out_path = out_dir / f"img2table_pages_{start + 1}-{end}.xlsx"
        extracted = pdf.extract_tables(ocr=ocr, borderless_tables=args.borderless)
        pdf.to_xlsx(dest=str(out_path), ocr=ocr, borderless_tables=args.borderless)

        num_pages = len(extracted)
        num_tables = sum(len(v) for v in extracted.values())
        print(f"Batch {start + 1}-{end}: pages with tables {num_pages}, total tables {num_tables}")
        print(f"Output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
