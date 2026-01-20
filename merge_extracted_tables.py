#!/usr/bin/env python3
"""
Merge extracted table JSONs (from extract_tables.py) across PDFs by page + header signature.

Outputs merged CSVs with an added source PDF column.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import calendar
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def table_to_grid(table: dict) -> List[List[str]]:
    rows = table["rowCount"]
    cols = table["columnCount"]
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in table.get("cells", []):
        r = cell.get("rowIndex", 0)
        c = cell.get("columnIndex", 0)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = cell.get("content", "")
    return grid


DATE_PATTERNS = [
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    for pat in DATE_PATTERNS:
        text = re.sub(pat, "<date>", text)
    # Remove digits and common date separators to stabilize signatures.
    text = re.sub(r"[0-9./-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.replace(" ", "")


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def header_rows_from_cells(table: dict) -> List[int]:
    rows = sorted({c.get("rowIndex", 0) for c in table.get("cells", []) if c.get("kind") == "columnHeader"})
    if rows:
        return rows
    # Fallback when kind is missing or unreliable.
    row_count = table.get("rowCount", 0)
    return [r for r in range(min(2, row_count))]


def prune_header_rows(grid: List[List[str]], header_rows: List[int]) -> List[int]:
    pruned = []
    for r in header_rows:
        if 0 <= r < len(grid):
            normalized = [normalize_text(v) for v in grid[r] if v.strip()]
            non_empty = len(normalized)
            if non_empty >= 2:
                # Skip rows that are only short tokens like R/D.
                if all(len(re.sub(r"[^a-z]+", "", v)) <= 2 for v in normalized):
                    continue
                pruned.append(r)
    return pruned


def extract_header_context(grid: List[List[str]], header_rows: List[int]) -> str:
    if not grid:
        return ""
    if not header_rows:
        header_rows = list(range(min(3, len(grid))))

    lines = []
    for r in header_rows:
        if 0 <= r < len(grid):
            row_text = " ".join(cell for cell in grid[r] if cell.strip())
            # Fix split dates like "31/01/ 2018" -> "31/01/2018".
            row_text = re.sub(r"(\d{1,2}[./-]\d{1,2}[./-])\s+(\d{2,4})", r"\\1\\2", row_text)
            row_text = re.sub(r"\s+", " ", row_text).strip()
            if row_text:
                lines.append(row_text)

    if not lines:
        return ""
    return " | ".join(lines)


def extract_header_periods_from_context(context: str) -> str:
    if not context:
        return ""
    blob = context
    date = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    ranges = re.findall(rf"({date})\s*(?:to|to\.|-)\s*({date})", blob, flags=re.I)
    if ranges:
        seen = set()
        out = []
        for start, end in ranges:
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{start}-{end}")
        return " | ".join(out)

    dates = re.findall(date, blob)
    if dates:
        seen = set()
        out = []
        for d in dates:
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
        return " | ".join(out)
    return ""


def clean_header_label(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", "", text)
    text = re.sub(r"\bto\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_rd_token(text: str) -> bool:
    return text.strip().upper() in ("R", "D")


def build_merged_header(grid: List[List[str]], header_rows: List[int]) -> List[str]:
    if not grid:
        return []
    cols = len(grid[0])
    base_labels = [""] * cols
    rd_labels = [""] * cols

    for c in range(cols):
        parts: List[str] = []
        for r in header_rows:
            if 0 <= r < len(grid):
                cell = grid[r][c].strip()
                if not cell:
                    continue
                if is_rd_token(cell):
                    rd_labels[c] = cell.strip().upper()
                    continue
                cleaned = clean_header_label(cell)
                if cleaned and not re.search(r"\d", cleaned):
                    parts.append(cleaned)
        if parts:
            base_labels[c] = " ".join(parts).strip()

    last = ""
    for c in range(cols):
        if base_labels[c]:
            last = base_labels[c]
        else:
            base_labels[c] = last

    labels = []
    for c in range(cols):
        label = base_labels[c].strip()
        if rd_labels[c]:
            label = f"{label} {rd_labels[c]}".strip()
        labels.append(label)
    return labels


def extract_label_ranges(grid: List[List[str]], header_rows: List[int]) -> Dict[str, str]:
    if not grid:
        return {}
    if not header_rows:
        header_rows = list(range(min(3, len(grid))))

    labels = {
        "current_month": ["current month"],
        "previous_month": ["previous month"],
        "current_year": ["current year"],
        "previous_year": ["previous year"],
    }

    cols = len(grid[0])
    results: Dict[str, str] = {}

    for key, phrases in labels.items():
        found_range = ""
        for c in range(cols):
            col_label_text = " ".join(
                normalize_label(grid[r][c]) for r in header_rows if 0 <= r < len(grid)
            )
            if not col_label_text:
                continue
            if not any(p in col_label_text for p in phrases):
                continue

            dates: List[str] = []
            for r in header_rows:
                if 0 <= r < len(grid):
                    cell = grid[r][c]
                    cell = re.sub(
                        r"(\d{1,2}[./-]\d{1,2}[./-])\s+(\d{2,4})", r"\\1\\2", cell
                    )
                    date = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
                    ranges = re.findall(
                        rf"({date})\s*(?:to|to\.|-)\s*({date})",
                        cell,
                        flags=re.I,
                    )
                    if ranges:
                        start, end = ranges[0]
                        found_range = f"{start}-{end}"
                        break
                    dates.extend(re.findall(date, cell))

            if found_range:
                break
            if len(dates) >= 2:
                found_range = f"{dates[0]}-{dates[-1]}"
                break
            if len(dates) == 1:
                found_range = dates[0]
                break

        results[key] = found_range

    return results


def parse_date(value: str) -> date | None:
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", value.strip())
    if not m:
        return None
    day, month, year = map(int, m.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def format_date(value: date, sep: str) -> str:
    return f"{value.day:02d}{sep}{value.month:02d}{sep}{value.year:04d}"


def end_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def shift_year_safe(value: date, years: int) -> date:
    year = value.year + years
    day = min(value.day, end_of_month(year, value.month))
    return date(year, value.month, day)


def derive_ranges_from_current(current_range: str) -> Dict[str, str]:
    m = re.match(
        r"^\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*-\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*$",
        current_range,
    )
    if not m:
        return {}
    start_raw, end_raw = m.groups()
    start = parse_date(start_raw)
    end = parse_date(end_raw)
    if not start or not end:
        return {}
    sep = "." if "." in current_range else ("/" if "/" in current_range else "-")

    # Previous month range.
    if start.month == 1:
        prev_year = start.year - 1
        prev_month = 12
    else:
        prev_year = start.year
        prev_month = start.month - 1
    prev_start = date(prev_year, prev_month, 1)
    prev_end = date(prev_year, prev_month, end_of_month(prev_year, prev_month))

    # Current year range (from Jan 1 to current month end).
    curr_year_start = date(start.year, 1, 1)
    curr_year_end = end

    # Previous year range (from Jan 1 to same month/day last year).
    prev_year_start = date(start.year - 1, 1, 1)
    prev_year_end = shift_year_safe(end, -1)

    return {
        "current_month": f"{format_date(start, sep)}-{format_date(end, sep)}",
        "previous_month": f"{format_date(prev_start, sep)}-{format_date(prev_end, sep)}",
        "current_year": f"{format_date(curr_year_start, sep)}-{format_date(curr_year_end, sep)}",
        "previous_year": f"{format_date(prev_year_start, sep)}-{format_date(prev_year_end, sep)}",
    }


def month_year_range_from_text(text: str) -> str:
    date_pat = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    matches = re.findall(date_pat, text)
    if not matches:
        return text
    dates = [parse_date(m) for m in matches]
    dates = [d for d in dates if d]
    if not dates:
        return text
    start = dates[0]
    end = dates[-1]
    start_val = f"{start.month:02d}-{start.year:04d}"
    end_val = f"{end.month:02d}-{end.year:04d}"
    if start_val == end_val:
        return start_val
    return f"{start_val} to {end_val}"


def collapse_range(text: str, mode: str) -> str:
    date_pat = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    matches = re.findall(date_pat, text)
    if not matches:
        return text
    dates = [parse_date(m) for m in matches]
    dates = [d for d in dates if d]
    if not dates:
        return text
    end = dates[-1]
    if mode == "month":
        return f"{end.month:02d}-{end.year:04d}"
    if mode == "year":
        return f"{end.year:04d}"
    return text


def header_signature(grid: List[List[str]], header_rows: List[int]) -> str:
    if not grid:
        return ""
    cols = len(grid[0])
    # Only include columns with any header text to avoid empty padding columns.
    header_cols = []
    for c in range(cols):
        has_text = False
        for r in header_rows:
            if 0 <= r < len(grid):
                if normalize_text(grid[r][c]):
                    has_text = True
                    break
        if has_text:
            header_cols.append(c)

    parts = []
    for c in header_cols:
        col_parts = []
        for r in header_rows:
            if 0 <= r < len(grid):
                val = normalize_text(grid[r][c])
                if val:
                    col_parts.append(val)
        parts.append("|".join(col_parts))
    return "||".join(parts).strip("|")


def data_rows(grid: List[List[str]], header_rows: Iterable[int]) -> List[List[str]]:
    header_set = set(header_rows)
    rows = []
    for r, row in enumerate(grid):
        if r in header_set:
            continue
        if any(cell.strip() for cell in row):
            rows.append(row)
    return rows


def source_name_for_path(path: Path) -> str:
    if path.parent.name and path.parent.name != "extracted_tables":
        return path.parent.name
    return path.stem


def write_csv(rows: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        for row in rows:
            f.write(",".join(escape_csv(cell) for cell in row))
            f.write("\n")


def escape_csv(cell: str) -> str:
    if any(ch in cell for ch in [",", "\"", "\n"]):
        return "\"" + cell.replace("\"", "\"\"") + "\""
    return cell


def slugify(text: str, limit: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:limit] or "table"


def merge_tables(
    input_paths: List[Path],
    outdir: Path,
    source_col: str,
) -> None:
    merged: Dict[Tuple[Tuple[int, ...], str], dict] = {}

    for path in input_paths:
        doc = load_json(path)
        source = source_name_for_path(path)
        for table in doc:
            pages = tuple(table.get("pageNumbers", []))
            grid = table_to_grid(table)
            header_rows_all = header_rows_from_cells(table)
            header_rows = prune_header_rows(grid, header_rows_all)
            sig = header_signature(grid, header_rows)
            label_ranges = extract_label_ranges(grid, header_rows)
            derived = derive_ranges_from_current(label_ranges.get("current_month", ""))
            if derived:
                label_ranges.update(derived)
            if label_ranges.get("current_month"):
                label_ranges["current_month"] = collapse_range(
                    label_ranges["current_month"], "month"
                )
            if label_ranges.get("previous_month"):
                label_ranges["previous_month"] = collapse_range(
                    label_ranges["previous_month"], "month"
                )
            if label_ranges.get("current_year"):
                label_ranges["current_year"] = collapse_range(
                    label_ranges["current_year"], "month"
                )
            if label_ranges.get("previous_year"):
                label_ranges["previous_year"] = collapse_range(
                    label_ranges["previous_year"], "month"
                )
            # Coarse grouping for known tables with unstable headers.
            if "propertyinvolved" in sig:
                sig = "eow_property_involved"
            elif "propertyrecovered" in sig:
                sig = "eow_property_recovered"
            key = (pages, sig)

            entry = merged.setdefault(
                key,
                {
                    "pages": pages,
                    "signature": sig,
                    "header_labels": build_merged_header(grid, header_rows_all),
                    "rows": [],
                    "columnCount": len(grid[0]) if grid else 0,
                },
            )

            for row in data_rows(grid, header_rows_all):
                col_count = entry["columnCount"]
                if len(row) > col_count:
                    # Expand existing header rows to match new max width.
                    diff = len(row) - col_count
                    entry["header_labels"] = entry["header_labels"] + [""] * diff
                    entry["columnCount"] = len(row)
                    col_count = entry["columnCount"]
                if len(row) < col_count:
                    row = row + [""] * (col_count - len(row))
                entry["rows"].append(
                    [
                        source,
                        label_ranges.get("current_month", ""),
                        label_ranges.get("previous_month", ""),
                        label_ranges.get("current_year", ""),
                        label_ranges.get("previous_year", ""),
                    ]
                    + row
                )

    # Write outputs
    outdir.mkdir(parents=True, exist_ok=True)
    for idx, (key, entry) in enumerate(merged.items(), 1):
        pages = entry["pages"] or ()
        page_part = "-".join(str(p) for p in pages) or "unknown"
        sig_part = slugify(entry["signature"])
        out_path = outdir / f"page_{page_part}_table_{idx}_{sig_part}.csv"

        header = entry["header_labels"]
        if header:
            header = (
                [
                    source_col,
                    "source_current_month",
                    "source_previous_month",
                    "source_current_year",
                    "source_previous_year",
                ]
                + header
            )
        rows = ([header] if header else []) + entry["rows"]
        write_csv(rows, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge extracted_tables.json files by page + header signature."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Paths to extracted_tables.json (default: extracted_tables/*/extracted_tables.json)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("merged_tables"))
    parser.add_argument("--source-col", default="source_pdf")
    args = parser.parse_args()

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = [Path(p) for p in glob.glob("extracted_tables/*/extracted_tables.json")]

    if not input_paths:
        raise SystemExit("No input files found.")

    merge_tables(input_paths, args.outdir, args.source_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
