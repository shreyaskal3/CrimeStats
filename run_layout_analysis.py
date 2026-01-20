import argparse
import json
import time
from pathlib import Path

import requests

from config import layout_key, layout_url


DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_POLL_TIMEOUT = 180.0


def submit_pdf(pdf_path):
    headers = {
        "Ocp-Apim-Subscription-Key": layout_key,
        "Content-Type": "application/pdf",
    }
    with open(pdf_path, "rb") as f:
        resp = requests.post(layout_url, headers=headers, data=f, timeout=60)
    resp.raise_for_status()
    operation_url = resp.headers.get("Operation-Location")
    if not operation_url:
        raise RuntimeError("Missing Operation-Location header from API response.")
    return operation_url


def poll_result(operation_url, poll_interval, poll_timeout):
    headers = {"Ocp-Apim-Subscription-Key": layout_key}
    deadline = time.time() + poll_timeout
    while True:
        resp = requests.get(operation_url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status in ("succeeded", "failed"):
            return data
        if time.time() >= deadline:
            raise TimeoutError(f"Polling timed out after {poll_timeout} seconds.")
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Run Azure Form Recognizer prebuilt-layout on a single PDF."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--out",
        help="Output JSON path (default: layout_outputs/<pdf_stem>.json)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between status checks",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=DEFAULT_POLL_TIMEOUT,
        help="Max seconds to wait for completion",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("layout_outputs") / f"{pdf_path.stem}.json"

    print(f"Submitting {pdf_path} ...")
    operation_url = submit_pdf(pdf_path)
    print("Polling for result ...")
    result = poll_result(operation_url, args.poll_interval, args.poll_timeout)

    status = result.get("status")
    if status != "succeeded":
        raise RuntimeError(f"Analysis failed with status: {status}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)
    print(f"Saved result: {out_path}")


if __name__ == "__main__":
    main()
