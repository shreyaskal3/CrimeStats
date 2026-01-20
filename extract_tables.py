#!/usr/bin/env python3
"""
Extract tables from Azure Document Intelligence (prebuilt-layout) JSON.

Rebuilds cell text from page words + cell bounding boxes to avoid
content stream concatenation issues (e.g., labels glued to numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def poly_bounds(polygon: List[float]) -> Tuple[float, float, float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def word_centers(page: dict) -> List[dict]:
    words = []
    for w in page.get("words", []):
        poly = w.get("polygon", [])
        if len(poly) != 8:
            continue
        minx, miny, maxx, maxy = poly_bounds(poly)
        w = dict(w)
        w["_center"] = ((minx + maxx) / 2, (miny + maxy) / 2)
        words.append(w)
    return words


def words_in_bounds(words: Iterable[dict], bounds: Tuple[float, float, float, float]) -> List[dict]:
    minx, miny, maxx, maxy = bounds
    picked = []
    for w in words:
        cx, cy = w["_center"]
        if minx <= cx <= maxx and miny <= cy <= maxy:
            picked.append(w)
    return picked


def join_words(words: List[dict], y_tol: float = 0.02) -> str:
    if not words:
        return ""
    # Group into lines by y tolerance, then sort each line by x.
    by_y = sorted(words, key=lambda w: w["_center"][1])
    lines: List[List[dict]] = []
    current = [by_y[0]]
    current_y = by_y[0]["_center"][1]
    for w in by_y[1:]:
        if abs(w["_center"][1] - current_y) <= y_tol:
            current.append(w)
            current_y = (current_y + w["_center"][1]) / 2
        else:
            lines.append(current)
            current = [w]
            current_y = w["_center"][1]
    lines.append(current)

    parts = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["_center"][0])
        parts.append(" ".join(w["content"] for w in line_sorted))
    return " ".join(parts)


def split_trailing_digits(text: str) -> Tuple[str, str] | None:
    m = re.match(r"^(.*?)(\d+)$", text.strip())
    if not m:
        return None
    left = m.group(1).rstrip()
    digits = m.group(2)
    if not left or not any(ch.isalpha() for ch in left):
        return None
    return left, digits


def fix_trailing_digits(grid: List[List[str]]) -> None:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols - 1):
            if not grid[r][c] or grid[r][c + 1]:
                continue
            split = split_trailing_digits(grid[r][c])
            if not split:
                continue
            left, digits = split
            grid[r][c] = left
            grid[r][c + 1] = digits


def extract_tables(doc: dict) -> List[dict]:
    analyze = doc.get("analyzeResult", {})
    pages = {p["pageNumber"]: p for p in analyze.get("pages", [])}
    words_by_page = {num: word_centers(p) for num, p in pages.items()}

    tables_out = []
    for t_idx, table in enumerate(analyze.get("tables", []), 1):
        page_nums = sorted({br["pageNumber"] for br in table.get("boundingRegions", [])})
        rows = table.get("rowCount", 0)
        cols = table.get("columnCount", 0)

        cells_out = []
        for cell in table.get("cells", []):
            content = ""
            cell_regions = cell.get("boundingRegions", [])
            for region in cell_regions:
                page_num = region.get("pageNumber")
                poly = region.get("polygon", [])
                if page_num in words_by_page and len(poly) == 8:
                    bounds = poly_bounds(poly)
                    words = words_in_bounds(words_by_page[page_num], bounds)
                    content = join_words(words)
                    break
            if not content:
                content = cell.get("content", "")

            cells_out.append(
                {
                    "rowIndex": cell.get("rowIndex", 0),
                    "columnIndex": cell.get("columnIndex", 0),
                    "rowSpan": cell.get("rowSpan", 1),
                    "columnSpan": cell.get("columnSpan", 1),
                    "kind": cell.get("kind", "content"),
                    "content": content,
                }
            )

        table_out = {
            "tableIndex": t_idx,
            "pageNumbers": page_nums,
            "rowCount": rows,
            "columnCount": cols,
            "cells": cells_out,
        }

        # Fix common OCR issue: digits stuck to the end of text cells.
        grid = table_to_grid(table_out)
        fix_trailing_digits(grid)
        cell_map = {(c["rowIndex"], c["columnIndex"]): c for c in cells_out}
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if (r, c) in cell_map:
                    cell_map[(r, c)]["content"] = val
                elif val:
                    cells_out.append(
                        {
                            "rowIndex": r,
                            "columnIndex": c,
                            "rowSpan": 1,
                            "columnSpan": 1,
                            "kind": "content",
                            "content": val,
                        }
                    )

        tables_out.append(table_out)

    return tables_out


def table_to_grid(table: dict) -> List[List[str]]:
    rows = table["rowCount"]
    cols = table["columnCount"]
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in table["cells"]:
        r = cell["rowIndex"]
        c = cell["columnIndex"]
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = cell["content"]
    return grid


def write_csv(grid: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract tables from layout JSON.")
    parser.add_argument("input", type=Path, help="Path to layout JSON file")
    parser.add_argument("--outdir", type=Path, default=Path("extracted_tables"))
    parser.add_argument("--no-csv", action="store_true", help="Do not emit CSVs")
    parser.add_argument("--no-json", action="store_true", help="Do not emit JSON")
    args = parser.parse_args()

    doc = load_json(args.input)
    tables = extract_tables(doc)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_json:
        json_path = args.outdir / "extracted_tables.json"
        json_path.write_text(json.dumps(tables, indent=2))

    if not args.no_csv:
        for table in tables:
            pages = "-".join(str(p) for p in table["pageNumbers"]) or "unknown"
            name = f"table_{table['tableIndex']}_page_{pages}.csv"
            grid = table_to_grid(table)
            write_csv(grid, args.outdir / name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
