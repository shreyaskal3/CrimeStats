import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pdfplumber


NUM_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*$|^\d+$")
EXPORT_PAGE_DEBUG = True
DEBUG_DIR = Path("page_debug")
USE_PIXEL_LINES = True
USE_WORD_COLUMNS = True
EXPORT_GRID_DEBUG = True
EXTEND_GRID_BELOW = False
MIN_ROW_GAP_RATIO = 0.6
MIN_ROW_GAP_PX = 6
MIN_COL_GAP_RATIO = 0.3
MIN_COL_GAP_PX = 1
COL_WHITE_THRESHOLD_RATIO = 0.09
COL_SMOOTH_WINDOW = 1
COL_BAND_WIDTH = 200


def normalize_cell(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def is_numeric_token(token):
    return bool(NUM_TOKEN_RE.fullmatch(token))


def smart_expand_rows(rows):
    expanded = []
    for row in rows:
        split_cells = [[p.strip() for p in str(c or "").split("\n")] for c in row]
        counts = [len(parts) for parts in split_cells]
        max_len = max(counts) if counts else 1
        multi_cells = sum(1 for c in counts if c > 1)
        if max_len <= 1 or multi_cells <= 1:
            merged = [normalize_cell(" ".join(parts)) for parts in split_cells]
            expanded.append(merged)
            continue
        for i in range(max_len):
            new_row = []
            for parts in split_cells:
                new_row.append(normalize_cell(parts[i]) if i < len(parts) else "")
            expanded.append(new_row)
    return expanded


def cluster_positions(values, tolerance=2):
    values = sorted(values)
    clusters = []
    for val in values:
        if not clusters or abs(val - clusters[-1][-1]) > tolerance:
            clusters.append([val])
        else:
            clusters[-1].append(val)
    return [sum(c) / len(c) for c in clusters]


def group_words_by_line(words, y_tolerance=2):
    lines = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for line in lines:
            if abs(word["top"] - line["top"]) <= y_tolerance:
                line["words"].append(word)
                line["top"] = (line["top"] * line["count"] + word["top"]) / (line["count"] + 1)
                line["count"] += 1
                placed = True
                break
        if not placed:
            lines.append({"top": word["top"], "count": 1, "words": [word]})
    for line in lines:
        line["words"] = sorted(line["words"], key=lambda w: w["x0"])
    return lines


def merge_hyphenated_words(words, max_gap=5):
    merged = []
    idx = 0
    while idx < len(words):
        current = dict(words[idx])
        if current["text"].endswith("-") and idx + 1 < len(words):
            nxt = words[idx + 1]
            gap = nxt["x0"] - current["x1"]
            if gap >= 0 and gap <= max_gap:
                current["text"] = current["text"][:-1] + nxt["text"]
                current["x1"] = nxt["x1"]
                idx += 2
                merged.append(current)
                continue
        merged.append(current)
        idx += 1
    return merged


def estimate_row_height_from_lines(lines):
    data_lines = []
    for line in lines:
        if not line["words"]:
            continue
        first = line["words"][0]["text"]
        if re.fullmatch(r"\d+", first):
            data_lines.append(line)
    if len(data_lines) < 3:
        return None
    tops = sorted(line["top"] for line in data_lines)
    diffs = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    diffs = [d for d in diffs if 4 <= d <= 40]
    if not diffs:
        return None
    return float(np.median(diffs))


def estimate_table_bottom_from_lines(lines):
    data_lines = []
    for line in lines:
        if not line["words"]:
            continue
        first = line["words"][0]["text"]
        if re.fullmatch(r"\d+", first):
            data_lines.append(line)
    if not data_lines:
        return None
    return max(max(w["bottom"] for w in line["words"]) for line in data_lines)


def estimate_data_top_from_lines(lines):
    data_lines = []
    for line in lines:
        if not line["words"]:
            continue
        first = line["words"][0]["text"]
        if re.fullmatch(r"\d+", first):
            data_lines.append(line)
    if not data_lines:
        return None
    return min(min(w["top"] for w in line["words"]) for line in data_lines)


def compute_row_boundaries(lines):
    data_lines = []
    for line in lines:
        if not line["words"]:
            continue
        first = line["words"][0]["text"]
        if re.fullmatch(r"\d+", first):
            data_lines.append(line)
    if len(data_lines) < 2:
        return []
    tops = sorted(line["top"] for line in data_lines)
    diffs = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    diffs = [d for d in diffs if 4 <= d <= 40]
    if not diffs:
        return []
    row_h = float(np.median(diffs))
    start = tops[0] - row_h / 2
    boundaries = [start]
    for i in range(len(tops) - 1):
        boundaries.append((tops[i] + tops[i + 1]) / 2)
    boundaries.append(tops[-1] + row_h / 2)
    return boundaries


def compute_col_gap_centers(bw, x_left_px, x_right_px, y_top_px, y_bottom_px):
    crop = bw[y_top_px:y_bottom_px, x_left_px:x_right_px]
    if crop.size == 0:
        return []

    col_proj = crop.sum(axis=0) / 255
    band = max(1, int(COL_BAND_WIDTH))
    if band > 1:
        band_kernel = np.ones(band) / band
        col_proj = np.convolve(col_proj, band_kernel, mode="same")
    window = max(5, int(COL_SMOOTH_WINDOW))
    kernel = np.ones(window) / window
    smooth = np.convolve(col_proj, kernel, mode="same")
    thr = COL_WHITE_THRESHOLD_RATIO * crop.shape[0]
    mask = smooth < thr

    segs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))

    segs = [s for s in segs if (s[1] - s[0] + 1) >= MIN_COL_GAP_PX]
    centers = [int((s[0] + s[1]) / 2) + x_left_px for s in segs]

    if len(centers) > 1:
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        diffs = [d for d in diffs if d > 0]
        if diffs:
            median_gap = float(np.median(diffs))
            min_gap = max(MIN_COL_GAP_PX, median_gap * MIN_COL_GAP_RATIO)
            filtered = []
            for c in centers:
                if not filtered or (c - filtered[-1]) >= min_gap:
                    filtered.append(c)
            centers = filtered

    return centers

def derive_column_boundaries_from_words(lines, page_width, tolerance=8):
    data_lines = []
    for line in lines:
        num_count = sum(1 for w in line["words"] if re.search(r"\d", w["text"]))
        if num_count >= 4 or (line["words"] and line["words"][0]["text"].isdigit()):
            data_lines.append(line)

    if not data_lines:
        return []

    numeric_words = []
    sign_words = []
    alpha_words = []
    for line in data_lines:
        for w in line["words"]:
            text = w["text"]
            if text in {"+", "-"}:
                sign_words.append(w)
            if re.search(r"\d", text):
                numeric_words.append(w)
            if re.search(r"[A-Za-z]", text):
                alpha_words.append(w)

    if not numeric_words:
        return []

    sr_candidates = [w for w in numeric_words if w["x0"] < 160]
    sr_x1_max = max((w["x1"] for w in sr_candidates), default=min(w["x0"] for w in numeric_words))

    first_numeric_candidates = []
    for line in data_lines:
        nums = sorted([w for w in line["words"] if re.search(r"\d", w["text"])], key=lambda w: w["x0"])
        for w in nums:
            if w["x0"] > sr_x1_max + 20:
                first_numeric_candidates.append(w["x0"])
                break
    first_numeric_x0 = min(first_numeric_candidates) if first_numeric_candidates else sr_x1_max + 40

    crime_min_x0 = min((w["x0"] for w in alpha_words), default=sr_x1_max + 10)
    crime_max_x1 = max((w["x1"] for w in alpha_words if w["x1"] < first_numeric_x0), default=crime_min_x0)

    sr_crime_border = (sr_x1_max + crime_min_x0) / 2
    crime_numeric_border = (crime_max_x1 + first_numeric_x0) / 2

    numeric_centers = [
        (w["x0"] + w["x1"]) / 2 for w in numeric_words if w["x0"] >= first_numeric_x0 - 5
    ]
    sign_centers = [(w["x0"] + w["x1"]) / 2 for w in sign_words]
    centers = numeric_centers + sign_centers
    if not centers:
        return []

    centers = sorted(cluster_positions(centers, tolerance=tolerance))
    if len(centers) < 2:
        return []

    boundaries = []
    left = min(w["x0"] for w in numeric_words + alpha_words) - 2
    right = max(w["x1"] for w in numeric_words + alpha_words) + 2
    left = max(0, left)
    right = min(page_width, right)

    boundaries.append(left)
    boundaries.append(sr_crime_border)
    boundaries.append(crime_numeric_border)
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) / 2)
    boundaries.append(right)

    boundaries = sorted({round(b, 2) for b in boundaries})
    return boundaries


def build_table_from_words(page):
    words = page.extract_words() or []
    if not words:
        return []
    lines = group_words_by_line(words)
    boundaries = derive_column_boundaries_from_words(lines, page.width)
    if len(boundaries) < 3:
        return []

    rows = []
    for line in lines:
        tokens = merge_hyphenated_words(line["words"])
        cells = [""] * (len(boundaries) - 1)
        for word in tokens:
            center = (word["x0"] + word["x1"]) / 2
            col_idx = None
            for i in range(len(boundaries) - 1):
                if boundaries[i] <= center < boundaries[i + 1]:
                    col_idx = i
                    break
            if col_idx is None:
                continue
            text = word["text"]
            if cells[col_idx]:
                if re.match(r"^[,.)]$", text):
                    cells[col_idx] = f"{cells[col_idx]}{text}"
                else:
                    cells[col_idx] = f"{cells[col_idx]} {text}"
            else:
                cells[col_idx] = text
        if any(c.strip() for c in cells):
            rows.append([normalize_cell(c) for c in cells])
    return rows


def extend_grid_debug_image(page, pdf_name, pixel_lines=None, resolution=200):
    try:
        page_image = page.to_image(resolution=resolution)
        pil_image = page_image.original
    except Exception:
        return {}

    img = np.array(pil_image)
    if img.ndim == 3:
        base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scale_x = img.shape[1] / page.width
    scale_y = img.shape[0] / page.height

    v_lines_px = []
    h_lines_px = []
    if pixel_lines:
        v_lines_px = [int(x * scale_x) for x in pixel_lines.get("vertical", [])]
        h_lines_px = [int(y * scale_y) for y in pixel_lines.get("horizontal", [])]

    v_lines_px = sorted(set(v_lines_px))
    h_lines_px = sorted(set(h_lines_px))

    # Determine table bounds using pixel lines first, then pixel projections.
    x_left_px = None
    x_right_px = None
    if len(v_lines_px) >= 2:
        x_left_px = min(v_lines_px)
        x_right_px = max(v_lines_px)
    else:
        col_proj = bw.sum(axis=0) / 255
        col_thr = 0.02 * bw.shape[0]
        cols = np.where(col_proj > col_thr)[0]
        if len(cols) > 0:
            x_left_px = int(cols[0])
            x_right_px = int(cols[-1])

    y_top_px = None
    y_header_top_px = None
    y_bottom_px = None
    if len(h_lines_px) >= 2:
        y_header_top_px = h_lines_px[0]
        y_top_px = h_lines_px[1] if len(h_lines_px) >= 2 else h_lines_px[0]
        y_bottom_px = h_lines_px[-1]
    else:
        row_proj = bw.sum(axis=1) / 255
        row_thr = 0.02 * bw.shape[1]
        rows = np.where(row_proj > row_thr)[0]
        if len(rows) > 0:
            y_header_top_px = int(rows[0])
            y_top_px = int(rows[0])
            y_bottom_px = int(rows[-1])

    if x_left_px is None or x_right_px is None or y_top_px is None or y_bottom_px is None:
        return {}
    if y_header_top_px is None:
        y_header_top_px = y_top_px

    # Compute row centers using horizontal projection inside table bounds.
    crop = bw[y_top_px:y_bottom_px, x_left_px:x_right_px]
    proj = crop.sum(axis=1) / 255
    thr = 0.02 * crop.shape[1]
    mask = proj > thr
    centers = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            end = i - 1
            centers.append((start + end) / 2 + y_top_px)
            start = None
    if start is not None:
        centers.append((start + len(mask) - 1) / 2 + y_top_px)

    diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    diffs = [d for d in diffs if 4 <= d <= 80]
    row_h_px = float(np.median(diffs)) if diffs else 20.0
    min_gap = max(MIN_ROW_GAP_PX, row_h_px * MIN_ROW_GAP_RATIO)
    filtered_centers = []
    for c in centers:
        if not filtered_centers or (c - filtered_centers[-1]) >= min_gap:
            filtered_centers.append(c)
    centers = filtered_centers

    # Build row boundaries from centers.
    row_boundaries_px = []
    if centers:
        row_boundaries_px.append(int(centers[0] - row_h_px / 2))
        for i in range(len(centers) - 1):
            row_boundaries_px.append(int((centers[i] + centers[i + 1]) / 2))
        row_boundaries_px.append(int(centers[-1] + row_h_px / 2))
    else:
        total_rows = int((y_bottom_px - y_top_px) // row_h_px)
        for k in range(total_rows + 1):
            row_boundaries_px.append(int(y_top_px + k * row_h_px))

    margin_px = 5
    n_rows = int((img.shape[0] - margin_px - y_bottom_px) // row_h_px)
    if n_rows <= 0:
        n_rows = 0
    y_end_px = int(y_bottom_px + n_rows * row_h_px) if EXTEND_GRID_BELOW else y_bottom_px

    # Draw detected pixel lines (blue/cyan) for comparison.
    if pixel_lines:
        for x in pixel_lines.get("vertical", []):
            x_px = int(x * scale_x)
            cv2.line(base, (x_px, y_top_px), (x_px, y_end_px), (255, 0, 0), 1)
        for y in pixel_lines.get("horizontal", []):
            y_px = int(y * scale_y)
            cv2.line(base, (x_left_px, y_px), (x_right_px, y_px), (255, 255, 0), 1)

    # Draw vertical lines using whitespace gaps (green).
    col_gap_centers = compute_col_gap_centers(bw, x_left_px, x_right_px, y_header_top_px, y_bottom_px)
    col_lines_px = [x_left_px] + col_gap_centers + [x_right_px]
    col_lines_px = sorted(set(col_lines_px))
    if len(col_lines_px) > 1:
        filtered = [col_lines_px[0]]
        for x in col_lines_px[1:]:
            if (x - filtered[-1]) >= MIN_COL_GAP_PX:
                filtered.append(x)
        col_lines_px = filtered

    for x in col_lines_px:
        cv2.line(base, (x, y_header_top_px), (x, y_end_px), (0, 255, 0), 1)

    # Draw horizontal grid lines based on pixel projections (red).
    for y in row_boundaries_px:
        if y_top_px <= y <= y_bottom_px:
            cv2.line(base, (x_left_px, y), (x_right_px, y), (0, 0, 255), 1)

    if EXTEND_GRID_BELOW:
        for k in range(1, n_rows + 1):
            y = int(y_bottom_px + k * row_h_px)
            cv2.line(base, (x_left_px, y), (x_right_px, y), (0, 0, 255), 1)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEBUG_DIR / f"{Path(pdf_name).stem}_grid_extend.png"
    cv2.imwrite(str(out_path), base)

    return {
        "image_path": str(out_path),
        "x_left": round(x_left_px / scale_x, 2),
        "x_right": round(x_right_px / scale_x, 2),
        "header_top": round(y_header_top_px / scale_y, 2),
        "y_bottom": round(y_bottom_px / scale_y, 2),
        "data_top": round(y_top_px / scale_y, 2),
        "row_height": round(row_h_px / scale_y, 2),
        "extra_rows": n_rows if EXTEND_GRID_BELOW else 0,
    }


def detect_table_lines_from_image(page, pdf_name, resolution=200):
    try:
        page_image = page.to_image(resolution=resolution)
        pil_image = page_image.original
    except Exception:
        return {"vertical": [], "horizontal": [], "image_path": ""}

    img = np.array(pil_image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, img.shape[1] // 40), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, img.shape[0] // 40)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    h_positions = []
    v_positions = []
    min_h_len = img.shape[1] * 0.3
    min_v_len = img.shape[0] * 0.3

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_h_len:
            h_positions.append(y + h / 2)

    contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_v_len:
            v_positions.append(x + w / 2)

    h_positions = cluster_positions(h_positions, tolerance=2)
    v_positions = cluster_positions(v_positions, tolerance=2)

    scale_x = page.width / img.shape[1]
    scale_y = page.height / img.shape[0]
    h_pdf = [round(p * scale_y, 2) for p in h_positions]
    v_pdf = [round(p * scale_x, 2) for p in v_positions]

    return {"vertical": v_pdf, "horizontal": h_pdf, "image_path": ""}


def parse_compact_table(table, pdf_name):
    lines = []
    for row in table:
        for cell in row:
            if cell:
                lines.extend(str(cell).splitlines())

    records = []
    for line in lines:
        line = normalize_cell(line)
        if not re.match(r"^\d+\s+", line):
            continue
        tokens = line.split()
        sr_no = int(tokens[0])
        rest = tokens[1:]
        idx = next((i for i, t in enumerate(rest) if is_numeric_token(t)), None)
        if idx is None:
            continue
        crime_head = " ".join(rest[:idx]).strip()
        nums = rest[idx:]
        diff_sign = ""
        diff_val = ""
        if len(nums) >= 2 and nums[-2] in {"+", "-"} and is_numeric_token(nums[-1]):
            diff_sign = nums[-2]
            diff_val = nums[-1]
            nums = nums[:-2]
        nums = nums + [""] * (10 - len(nums))
        records.append(
            {
                "pdf_file": pdf_name,
                "sr_no": sr_no,
                "crime_head": crime_head,
                "current_month_reg": nums[0],
                "current_month_det": nums[1],
                "previous_month_reg": nums[2],
                "previous_month_det": nums[3],
                "current_year_reg": nums[4],
                "current_year_det": nums[5],
                "current_year_det_pct": nums[6],
                "previous_year_reg": nums[7],
                "previous_year_det": nums[8],
                "previous_year_det_pct": nums[9],
                "diff_reg_sign": diff_sign,
                "diff_reg_value": diff_val,
            }
        )
    return records


def parse_table_rows(table, pdf_name):
    rows = smart_expand_rows(table)
    max_len = max(len(r) for r in rows if r) if rows else 0
    if max_len <= 3:
        return parse_compact_table(table, pdf_name)

    records = []
    sr_counter = 0
    for row in rows:
        row = row + [""] * (max_len - len(row))
        crime_head = normalize_cell(row[1]) if len(row) > 1 else ""
        if not crime_head:
            continue
        if re.search(r"crime heads?", crime_head, flags=re.I):
            continue
        sr_text = normalize_cell(row[0])
        if sr_text.isdigit():
            sr_counter = int(sr_text)
            sr_no = sr_counter
        else:
            sr_no = None
            if any(normalize_cell(c) for c in row[2:]):
                sr_counter += 1
                sr_no = sr_counter

        if max_len >= 14:
            diff_sign = normalize_cell(row[12])
            diff_val = normalize_cell(row[13])
        else:
            diff_sign = ""
            diff_val = ""

        records.append(
            {
                "pdf_file": pdf_name,
                "sr_no": sr_no,
                "crime_head": crime_head,
                "current_month_reg": normalize_cell(row[2]),
                "current_month_det": normalize_cell(row[3]),
                "previous_month_reg": normalize_cell(row[4]),
                "previous_month_det": normalize_cell(row[5]),
                "current_year_reg": normalize_cell(row[6]),
                "current_year_det": normalize_cell(row[7]),
                "current_year_det_pct": normalize_cell(row[8]),
                "previous_year_reg": normalize_cell(row[9]),
                "previous_year_det": normalize_cell(row[10]),
                "previous_year_det_pct": normalize_cell(row[11]),
                "diff_reg_sign": diff_sign if diff_sign in {"+", "-"} else diff_sign,
                "diff_reg_value": diff_val,
            }
        )
    return records


def parse_first_page(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return []
        page = pdf.pages[0]
        pixel_lines = {}
        if USE_PIXEL_LINES:
            pixel_lines = detect_table_lines_from_image(page, Path(pdf_path).name)
        grid_debug = {}
        if EXPORT_GRID_DEBUG:
            grid_debug = extend_grid_debug_image(page, Path(pdf_path).name, pixel_lines=pixel_lines)
        if EXPORT_PAGE_DEBUG:
            export_page_debug(page, Path(pdf_path).name, pixel_lines=pixel_lines, grid_debug=grid_debug)

        tables = page.extract_tables()
        if not tables and pixel_lines.get("vertical") and pixel_lines.get("horizontal"):
            table_settings = {
                "vertical_strategy": "explicit",
                "horizontal_strategy": "explicit",
                "explicit_vertical_lines": pixel_lines["vertical"],
                "explicit_horizontal_lines": pixel_lines["horizontal"],
                "intersection_tolerance": 5,
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }
            table = page.extract_table(table_settings)
            if table:
                return parse_table_rows(table, Path(pdf_path).name)

        if not tables and USE_WORD_COLUMNS:
            word_table = build_table_from_words(page)
            if word_table:
                return parse_table_rows(word_table, Path(pdf_path).name)
            return []
        if not tables:
            return []
        table = max(tables, key=lambda t: len(t) * max(len(r) for r in t if r))
        return parse_table_rows(table, Path(pdf_path).name)


def export_page_debug(page, pdf_name, pixel_lines=None, grid_debug=None):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "pdf_file": pdf_name,
        "page_index": 0,
        "width": page.width,
        "height": page.height,
        "text": page.extract_text() or "",
        "tables": page.extract_tables() or [],
        "words": page.extract_words() or [],
        "pixel_lines": pixel_lines or {},
        "grid_debug": grid_debug or {},
    }
    out_path = DEBUG_DIR / f"{Path(pdf_name).stem}_page1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def write_outputs(records, csv_path=None, json_path=None):
    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0].keys()) if records else [],
            )
            if records:
                writer.writeheader()
                writer.writerows(records)
    if json_path:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


def main(pdf_path=None, csv_path="parsed_first_page.csv", json_path="parsed_first_page.json"):
    if pdf_path is None:
        pdf_dir = Path("crime_pdfs")
        candidates = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        if not candidates:
            print("No PDF provided and none found in crime_pdfs/.")
            return
        pdf_path = candidates[0]

    records = parse_first_page(pdf_path)
    if not records:
        print("No rows parsed from the first page.")
        return
    write_outputs(records, csv_path=csv_path, json_path=json_path)
    print(f"Parsed {len(records)} rows from {pdf_path}")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    pdf_path = "crime_pdfs/1.pdf"
    main(pdf_path)
