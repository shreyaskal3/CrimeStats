#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

try:
    from config import OLLAMA_URL as DEFAULT_OLLAMA_URL
except Exception:
    DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """You are a data extraction engine. Convert the given plain-text table into a SINGLE valid JSON object.
Rules:
- Output MUST be valid JSON only. No markdown. No explanations. No comments.
- Preserve the exact numbers from input (including commas) but in JSON store numbers as integers (remove commas).
- If a number is missing in the table, use 0 for that numeric field unless clearly not applicable; use null only when a field truly does not exist (e.g., sr_no for Total row can be null).
- Keep crime/drug names exactly as they appear, but trim extra spaces and fix obvious broken spacing (e.g., "Cas es" -> "Cases", "Qt y" -> "Qty", "Val ue" -> "Value"). Do NOT invent new categories.
- Detect whether the table is an IPC comparative table or an NDPS comparative table:
  A) IPC table contains columns like Current Month/Previous Month/Current Year/Previous Year and R/D and % detection and Difference.
  B) NDPS table contains drug blocks (Heroin/Charas/Cocaine/Ganja) with Cases, P.A, Qty (Kg/Gram/Mili Gram), Value.
- Follow the schema exactly:
  - For IPC: include keys: title, jurisdiction, periods, columns, rows
  - For NDPS: include keys: title, scope, columns, rows
- For IPC: each row object must include sr_no, crime_head, current_month{R,D}, previous_month{R,D}, current_year{R,D,pct_detection_given,pct_detection_computed}, previous_year{R,D,pct_detection_given,pct_detection_computed}, difference_in_reg{sign,value,computed_current_minus_previous_year_R}
  pct_detection_computed = round((D/R)*100, 2) if R>0 else 0
- For NDPS: each row object must include location and drug keys: heroin, charas, cocaine, ganja
  each drug object: cases, pa, qty{kg,gram,milli_gram}, value
- Use integers for all numeric fields except pct_detection_computed which can be a number with decimals.
- Do not drop the Total row in IPC if present.
Return only the JSON object."""


def is_chat_endpoint(url: str) -> bool:
    return url.rstrip("/").endswith("/api/chat")


def extract_pages_from_pdf(pdf_path: Path) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF unavailable: {exc}") from exc

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    return pages


def extract_pages_from_text_file(text_path: Path) -> list[str]:
    text = text_path.read_text(encoding="utf-8", errors="replace")
    pages = text.split("\f")
    return pages if pages else [text]


def build_prompt(page_text: str) -> str:
    return f"{PROMPT}\n\nInput:\n{page_text.strip()}\n"


def call_ollama(url: str, model: str, prompt: str, timeout: int) -> str:
    payload: dict[str, object] = {
        "model": model,
        "stream": False,
    }
    if is_chat_endpoint(url):
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))

    if isinstance(data, dict) and "response" in data:
        return str(data["response"])
    if isinstance(data, dict) and isinstance(data.get("message"), dict):
        return str(data["message"].get("content", ""))

    return json.dumps(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send PDF text to Ollama one page at a time.",
    )
    parser.add_argument(
        "--text",
        default="parsed_outputs/1_pdftotext.txt",
        help="Path to the text file with form-feed page breaks.",
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="Optional PDF path to extract page-wise text.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama endpoint (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", ""),
        help="Ollama model name (or set OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--output-dir",
        default="ollama_outputs",
        help="Directory to write per-page outputs.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based start page (default: 1).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=0,
        help="1-based end page, inclusive (default: all).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Send empty pages as well.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model:
        print("ERROR: Missing model. Pass --model or set OLLAMA_MODEL.", flush=True)
        return 2

    pages: list[str] = []
    source = ""
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}", flush=True)
            return 2
        try:
            pages = extract_pages_from_pdf(pdf_path)
            source = str(pdf_path)
        except Exception as exc:
            print(f"PDF extraction failed, falling back to text: {exc}", flush=True)
    if not pages:
        text_path = Path(args.text)
        if not text_path.exists():
            print(f"ERROR: Text file not found: {text_path}", flush=True)
            return 2
        pages = extract_pages_from_text_file(text_path)
        source = str(text_path)

    total_pages = len(pages)
    if total_pages == 0:
        print(f"ERROR: No pages found in {source}", flush=True)
        return 2

    start_idx = max(args.start_page, 1) - 1
    end_idx = args.end_page if args.end_page > 0 else total_pages
    end_idx = min(end_idx, total_pages)

    if start_idx >= total_pages or start_idx >= end_idx:
        print("ERROR: Page range is empty.", flush=True)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using source: {source}")
    print(f"Sending pages {start_idx + 1} to {end_idx} of {total_pages}")
    print(f"Writing outputs to: {output_dir}")

    for page_num in range(start_idx + 1, end_idx + 1):
        page_text = pages[page_num - 1]
        if not args.keep_empty and not page_text.strip():
            print(f"Page {page_num}: skipped (empty)")
            continue

        prompt = build_prompt(page_text)
        try:
            response = call_ollama(args.url, args.model, prompt, args.timeout)
        except Exception as exc:
            err_path = output_dir / f"page_{page_num:03d}.error.txt"
            err_path.write_text(str(exc), encoding="utf-8")
            print(f"Page {page_num}: error (saved {err_path})")
            continue

        response_text = response.strip()
        try:
            json.loads(response_text)
            out_path = output_dir / f"page_{page_num:03d}.json"
        except json.JSONDecodeError:
            out_path = output_dir / f"page_{page_num:03d}.txt"
        out_path.write_text(response_text + "\n", encoding="utf-8")
        print(f"Page {page_num}: wrote {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
