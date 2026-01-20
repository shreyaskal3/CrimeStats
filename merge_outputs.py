import argparse
import json
import re
from pathlib import Path


DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


def month_label_from_date(date_str):
    if not date_str:
        return ""
    match = DATE_RE.search(date_str)
    if not match:
        return ""
    day, month, year = match.groups()
    return f"{year}-{month}"


def month_label_from_record(record):
    month = record.get("current_month")
    if month:
        return month
    return month_label_from_date(record.get("current_month_start", ""))


def iter_records(data, page_index):
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "page_index" in data[0] and "records" in data[0]:
            for page in data:
                if page_index is None or page.get("page_index") == page_index:
                    for record in page.get("records", []):
                        if isinstance(record, dict):
                            yield record
        else:
            for record in data:
                if isinstance(record, dict):
                    yield record
        return
    if isinstance(data, dict) and "records" in data:
        if page_index is None or data.get("page_index") == page_index:
            for record in data.get("records", []):
                if isinstance(record, dict):
                    yield record


def merge_outputs(input_dir, page_index):
    input_dir = Path(input_dir)
    merged = {}
    stats = {
        "files": 0,
        "records": 0,
        "kept": 0,
        "duplicates": 0,
        "skipped_missing_crime_head": 0,
        "skipped_missing_month": 0,
    }
    for json_path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stats["files"] += 1
        for record in iter_records(data, page_index):
            stats["records"] += 1
            crime_head = str(record.get("crime_head", "")).strip()
            if not crime_head:
                stats["skipped_missing_crime_head"] += 1
                continue
            month = month_label_from_record(record)
            if not month:
                stats["skipped_missing_month"] += 1
                continue
            month_map = merged.setdefault(crime_head, {})
            existing = month_map.get(month)
            if existing is None:
                month_map[month] = record
                stats["kept"] += 1
            elif isinstance(existing, list):
                existing.append(record)
                stats["duplicates"] += 1
            else:
                month_map[month] = [existing, record]
                stats["duplicates"] += 1
    return {"meta": stats, "data": merged}


def main():
    parser = argparse.ArgumentParser(description="Merge parsed_outputs JSON by crime_head and month.")
    parser.add_argument("--input-dir", default="parsed_outputs", help="Directory with per-PDF JSON outputs")
    parser.add_argument("--output", default="merged_outputs.json", help="Output JSON path")
    parser.add_argument(
        "--page-index",
        type=int,
        default=0,
        help="Only merge this page index (use --all-pages to ignore)",
    )
    parser.add_argument("--all-pages", action="store_true", help="Merge all pages")
    args = parser.parse_args()

    page_index = None if args.all_pages else args.page_index
    payload = merge_outputs(args.input_dir, page_index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
