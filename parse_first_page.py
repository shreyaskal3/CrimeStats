import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pdfplumber


NUM_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*$|^\d+$")
DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
EXPORT_PAGE_DEBUG = False
DEBUG_DIR = Path("page_debug")
USE_PIXEL_LINES = True
EXPORT_GRID_DEBUG = False
EXPORT_GAP_DEBUG = False
EXPORT_GAP_DEBUG_HEADER = False
EXPORT_OCR_OVERLAY = False
EXPORT_OCR_TABLE_ONLY = False
OCR_LINE_THICKNESS = 1
MIN_ROW_GAP_RATIO = 0.6
MIN_ROW_GAP_PX = 6
MIN_COL_GAP_RATIO = 0.3
MIN_COL_GAP_PX = 6
MIN_COL_GAP_PX_HEADER = 8
COL_WHITE_THRESHOLD_RATIO = 0.025
MIN_GAP_WHITE_RATIO = 0.9
GAP_SAMPLE_WIDTH_PX = 3
COL_SMOOTH_WINDOW = 4
COL_BAND_WIDTH = 10


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


def render_col_projection_plot(col_proj, smooth, thr):
    width = len(col_proj)
    if width <= 1:
        return None
    height = 200
    max_val = max(float(np.max(col_proj)), float(np.max(smooth)), float(thr), 1.0)
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_points(arr):
        ys = height - 1 - (arr / max_val * (height - 1))
        pts = np.column_stack((np.arange(width), ys)).astype(np.int32)
        return pts.reshape((-1, 1, 2))

    cv2.polylines(img, [to_points(col_proj)], False, (160, 160, 160), 1)
    cv2.polylines(img, [to_points(smooth)], False, (0, 0, 255), 1)
    thr_y = int(round(height - 1 - (thr / max_val) * (height - 1)))
    cv2.line(img, (0, thr_y), (width - 1, thr_y), (0, 200, 0), 1)
    return img


def export_gap_debug(
    crop,
    col_proj,
    smooth,
    thr,
    mask,
    segs,
    candidate_centers,
    candidate_white,
    accepted_centers,
    debug_prefix,
    x_left_px,
    y_top_px,
    min_gap_px,
):
    base = Path(debug_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.name

    crop_vis = 255 - crop
    cv2.imwrite(str(base.parent / f"{stem}_gap_01_crop.png"), crop_vis)

    proj_img = render_col_projection_plot(col_proj, smooth, thr)
    if proj_img is not None:
        cv2.imwrite(str(base.parent / f"{stem}_gap_02_projection.png"), proj_img)

    mask_row = np.where(mask, 255, 0).astype(np.uint8)
    mask_img = np.tile(mask_row, (crop.shape[0], 1))
    cv2.imwrite(str(base.parent / f"{stem}_gap_03_mask.png"), mask_img)

    cand_img = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
    for c in candidate_centers:
        x = int(round(c - x_left_px))
        if 0 <= x < cand_img.shape[1]:
            cv2.line(cand_img, (x, 0), (x, cand_img.shape[0] - 1), (0, 255, 255), 1)
    cv2.imwrite(str(base.parent / f"{stem}_gap_04_candidates.png"), cand_img)

    filt_img = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
    accepted_set = set(accepted_centers)
    for c in candidate_centers:
        x = int(round(c - x_left_px))
        if 0 <= x < filt_img.shape[1]:
            color = (0, 255, 0) if c in accepted_set else (0, 0, 255)
            cv2.line(filt_img, (x, 0), (x, filt_img.shape[0] - 1), color, 1)
    cv2.imwrite(str(base.parent / f"{stem}_gap_05_filtered.png"), filt_img)

    meta = {
        "crop": {
            "x_left": int(x_left_px),
            "x_right": int(x_left_px + crop.shape[1]),
            "y_top": int(y_top_px),
            "y_bottom": int(y_top_px + crop.shape[0]),
        },
        "thresholds": {
            "col_white_threshold": float(thr),
            "min_gap_white_ratio": float(MIN_GAP_WHITE_RATIO),
            "gap_sample_width_px": int(GAP_SAMPLE_WIDTH_PX),
            "min_col_gap_px": int(min_gap_px),
        },
        "segments": [
            {"start": int(s[0] + x_left_px), "end": int(s[1] + x_left_px)} for s in segs
        ],
        "candidates": [
            {
                "center_px": int(c),
                "white_ratio": round(float(candidate_white.get(c, 0.0)), 4),
                "accepted": c in accepted_set,
            }
            for c in candidate_centers
        ],
    }
    meta_path = base.parent / f"{stem}_gap_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)


def compute_col_gap_centers(
    bw,
    x_left_px,
    x_right_px,
    y_top_px,
    y_bottom_px,
    debug_prefix=None,
    min_gap_px=None,
):
    if min_gap_px is None:
        min_gap_px = MIN_COL_GAP_PX
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

    segs = [s for s in segs if (s[1] - s[0] + 1) >= min_gap_px]
    centers = [int((s[0] + s[1]) / 2) + x_left_px for s in segs]
    candidate_centers = list(centers)
    candidate_white = {}

    if centers:
        half = max(0, int(GAP_SAMPLE_WIDTH_PX) // 2)
        filtered = []
        for c in centers:
            idx = c - x_left_px
            start = max(0, idx - half)
            end = min(crop.shape[1], idx + half + 1)
            if end <= start:
                continue
            band = crop[:, start:end]
            white_ratio = np.count_nonzero(band == 0) / band.size
            candidate_white[c] = white_ratio
            if white_ratio >= MIN_GAP_WHITE_RATIO:
                filtered.append(c)
        centers = filtered

    if len(centers) > 1:
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        diffs = [d for d in diffs if d > 0]
        if diffs:
            median_gap = float(np.median(diffs))
            min_gap = max(min_gap_px, median_gap * MIN_COL_GAP_RATIO)
            filtered = []
            for c in centers:
                if not filtered or (c - filtered[-1]) >= min_gap:
                    filtered.append(c)
            centers = filtered

    if debug_prefix:
        export_gap_debug(
            crop,
            col_proj,
            smooth,
            thr,
            mask,
            segs,
            candidate_centers,
            candidate_white,
            centers,
            debug_prefix,
            x_left_px,
            y_top_px,
            min_gap_px,
        )

    return centers


def compute_whitespace_vertical_lines_from_image(
    page,
    pixel_lines=None,
    resolution=200,
    debug_prefix=None,
    debug_header_prefix=None,
):
    try:
        page_image = page.to_image(resolution=resolution)
        pil_image = page_image.original
    except Exception:
        return []

    img = np.array(pil_image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
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
        y_top_px = h_lines_px[1]
        y_bottom_px = h_lines_px[-1]
    elif len(h_lines_px) == 1:
        y_header_top_px = h_lines_px[0]
        y_top_px = h_lines_px[0]
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
        return [], []
    if x_right_px <= x_left_px or y_bottom_px <= y_top_px:
        return [], []

    col_gap_centers = compute_col_gap_centers(
        bw,
        x_left_px,
        x_right_px,
        y_top_px,
        y_bottom_px,
        debug_prefix=debug_prefix,
    )
    header_gap_centers = []
    if debug_header_prefix and y_header_top_px is not None:
        header_gap_centers = compute_col_gap_centers(
            bw,
            x_left_px,
            x_right_px,
            y_header_top_px,
            y_bottom_px,
            debug_prefix=debug_header_prefix,
            min_gap_px=MIN_COL_GAP_PX_HEADER,
        )
    data_lines = [round(c / scale_x, 2) for c in col_gap_centers] if col_gap_centers else []
    header_lines = [round(c / scale_x, 2) for c in header_gap_centers] if header_gap_centers else []
    return data_lines, header_lines

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


def extend_grid_debug_image(
    page,
    pdf_name,
    pixel_lines=None,
    gap_lines=None,
    header_gap_lines=None,
    resolution=200,
    page_index=0,
):
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
    y_end_px = y_bottom_px

    # Draw detected pixel lines (blue/cyan) for comparison.
    if pixel_lines:
        for x in pixel_lines.get("vertical", []):
            x_px = int(x * scale_x)
            cv2.line(base, (x_px, y_top_px), (x_px, y_end_px), (255, 0, 0), 1)
        for y in pixel_lines.get("horizontal", []):
            y_px = int(y * scale_y)
            cv2.line(base, (x_left_px, y_px), (x_right_px, y_px), (255, 255, 0), 1)

    # Draw vertical lines using whitespace gaps (green for data, orange for header-only).
    data_gap_px = []
    if gap_lines:
        data_gap_px = [
            int(x * scale_x)
            for x in gap_lines
            if x_left_px <= int(x * scale_x) <= x_right_px
        ]
    header_gap_px = []
    if header_gap_lines:
        header_gap_px = [
            int(x * scale_x)
            for x in header_gap_lines
            if x_left_px <= int(x * scale_x) <= x_right_px
        ]
    if gap_lines is None and not header_gap_lines:
        data_gap_px = compute_col_gap_centers(bw, x_left_px, x_right_px, y_top_px, y_bottom_px)

    # Draw table borders.
    cv2.line(base, (x_left_px, y_header_top_px), (x_left_px, y_end_px), (0, 255, 0), 1)
    cv2.line(base, (x_right_px, y_header_top_px), (x_right_px, y_end_px), (0, 255, 0), 1)

    for x in sorted(set(data_gap_px)):
        cv2.line(base, (x, y_top_px), (x, y_end_px), (0, 255, 0), 1)
    for x in sorted(set(header_gap_px)):
        cv2.line(base, (x, y_header_top_px), (x, y_top_px), (0, 165, 255), 1)

    # Draw horizontal grid lines based on pixel projections (red).
    for y in row_boundaries_px:
        if y_top_px <= y <= y_bottom_px:
            cv2.line(base, (x_left_px, y), (x_right_px, y), (0, 0, 255), 1)

    ocr_path = ""
    ocr_table_path = ""
    if EXPORT_OCR_OVERLAY:
        ocr = gray.copy()
        thickness = max(1, int(OCR_LINE_THICKNESS))
        for x in v_lines_px:
            if x_left_px <= x <= x_right_px:
                cv2.line(ocr, (x, y_top_px), (x, y_end_px), 0, thickness)
        cv2.line(ocr, (x_left_px, y_header_top_px), (x_left_px, y_end_px), 0, thickness)
        cv2.line(ocr, (x_right_px, y_header_top_px), (x_right_px, y_end_px), 0, thickness)
        for x in sorted(set(data_gap_px)):
            cv2.line(ocr, (x, y_top_px), (x, y_end_px), 0, thickness)
        for x in sorted(set(header_gap_px)):
            cv2.line(ocr, (x, y_header_top_px), (x, y_top_px), 0, thickness)
        for y in h_lines_px:
            cv2.line(ocr, (x_left_px, y), (x_right_px, y), 0, thickness)
        for y in row_boundaries_px:
            if y_top_px <= y <= y_bottom_px:
                cv2.line(ocr, (x_left_px, y), (x_right_px, y), 0, thickness)
        suffix = "" if page_index == 0 else f"_page{page_index + 1}"
        ocr_path = str(DEBUG_DIR / f"{Path(pdf_name).stem}{suffix}_ocr_overlay.png")
        cv2.imwrite(ocr_path, ocr)
        if EXPORT_OCR_TABLE_ONLY:
            x0 = max(0, int(x_left_px))
            x1 = min(ocr.shape[1], int(x_right_px) + 1)
            y0 = max(0, int(y_header_top_px))
            y1 = min(ocr.shape[0], int(y_end_px) + 1)
            if x1 > x0 and y1 > y0:
                ocr_table = ocr[y0:y1, x0:x1]
                ocr_table_path = str(
                    DEBUG_DIR / f"{Path(pdf_name).stem}{suffix}_ocr_overlay_table.png"
                )
                cv2.imwrite(ocr_table_path, ocr_table)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if page_index == 0 else f"_page{page_index + 1}"
    out_path = DEBUG_DIR / f"{Path(pdf_name).stem}{suffix}_grid_extend.png"
    cv2.imwrite(str(out_path), base)

    return {
        "image_path": str(out_path),
        "ocr_image_path": ocr_path,
        "ocr_table_image_path": ocr_table_path,
        "x_left": round(x_left_px / scale_x, 2),
        "x_right": round(x_right_px / scale_x, 2),
        "header_top": round(y_header_top_px / scale_y, 2),
        "y_bottom": round(y_bottom_px / scale_y, 2),
        "data_top": round(y_top_px / scale_y, 2),
        "row_height": round(row_h_px / scale_y, 2),
        "extra_rows": 0,
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


def row_tokens(row):
    tokens = []
    for cell in row:
        cell = normalize_cell(cell)
        if cell:
            tokens.extend(cell.split())
    return tokens


def estimate_header_bounds(page, pixel_lines):
    h_lines = sorted(pixel_lines.get("horizontal", [])) if pixel_lines else []
    if len(h_lines) >= 2:
        return h_lines[0], h_lines[1]
    if h_lines:
        return h_lines[0], h_lines[0] + page.height * 0.1
    return 0, page.height * 0.3


def extract_header_text(page, pixel_lines):
    words = page.extract_words() or []
    if not words:
        return ""
    y_header_top, y_top = estimate_header_bounds(page, pixel_lines)
    header_words = [
        w
        for w in words
        if (w["top"] >= y_header_top - 1) and (w["bottom"] <= y_top + 1)
    ]
    if not header_words:
        header_words = words
    lines = group_words_by_line(header_words)
    return "\n".join(" ".join(w["text"] for w in line["words"]) for line in lines)


def parse_month_label(date_str):
    try:
        day, month, year = date_str.split(".")
        return f"{year}-{month}"
    except ValueError:
        return ""


def build_header_date_tags(page, pixel_lines):
    header_text = extract_header_text(page, pixel_lines)
    dates = DATE_RE.findall(header_text)
    uniq = []
    for d in dates:
        if d not in uniq:
            uniq.append(d)
    dates = uniq[:8]
    if len(dates) < 8:
        fallback = DATE_RE.findall(page.extract_text() or "")
        uniq_fallback = []
        for d in fallback:
            if d not in uniq_fallback:
                uniq_fallback.append(d)
        if len(uniq_fallback) > len(dates):
            dates = uniq_fallback[:8]

    def pair(idx):
        start = dates[idx] if len(dates) > idx else ""
        end = dates[idx + 1] if len(dates) > idx + 1 else ""
        return start, end

    cur_start, cur_end = pair(0)
    prev_start, prev_end = pair(2)
    cur_year_start, cur_year_end = pair(4)
    prev_year_start, prev_year_end = pair(6)
    return {
        "current_month_start": cur_start,
        "current_month_end": cur_end,
        "current_month": parse_month_label(cur_start),
        "previous_month_start": prev_start,
        "previous_month_end": prev_end,
        "current_year_start": cur_year_start,
        "current_year_end": cur_year_end,
        "previous_year_start": prev_year_start,
        "previous_year_end": prev_year_end,
    }


def is_header_row(tokens):
    if not tokens:
        return False
    lowered = [t.lower().strip(".") for t in tokens]
    keywords = {
        "sr",
        "no",
        "crime",
        "heads",
        "current",
        "previous",
        "month",
        "year",
        "detection",
        "detec",
        "difference",
        "reg",
    }
    keyword_hits = sum(1 for t in lowered if any(k in t for k in keywords))
    numeric_count = sum(1 for t in tokens if is_numeric_token(t))
    if "sr" in lowered and "no" in lowered:
        return True
    if keyword_hits >= 3 and numeric_count <= 2:
        return True
    if keyword_hits >= 4:
        return True
    return False


def find_numeric_tail_index(tokens, min_nums=6, min_ratio=0.75):
    for i in range(len(tokens)):
        tail = tokens[i:]
        if not tail or not is_numeric_token(tail[0]):
            continue
        if len(tail) >= 2 and not (is_numeric_token(tail[1]) or tail[1] in {"+", "-"}):
            continue
        num_count = sum(1 for t in tail if is_numeric_token(t))
        sign_count = sum(1 for t in tail if t in {"+", "-"})
        ratio = (num_count + sign_count) / len(tail)
        if num_count >= min_nums and ratio >= min_ratio:
            return i
    return None


def parse_row_by_tokens(row, pdf_name):
    tokens = row_tokens(row)
    if not tokens:
        return None
    sr_no = None
    rest = tokens
    if tokens[0].isdigit():
        sr_no = int(tokens[0])
        rest = tokens[1:]
    idx = find_numeric_tail_index(rest)
    if idx is None:
        return None
    crime_head = " ".join(rest[:idx]).strip()
    if not crime_head:
        return None
    nums = rest[idx:]
    diff_sign = ""
    diff_val = ""
    if len(nums) >= 2 and nums[-2] in {"+", "-"} and is_numeric_token(nums[-1]):
        diff_sign = nums[-2]
        diff_val = nums[-1]
        nums = nums[:-2]
    nums = nums + [""] * (10 - len(nums))
    return {
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


def count_numeric_tokens(text):
    text = normalize_cell(text)
    if not text:
        return 0
    return sum(1 for t in text.split() if is_numeric_token(t))


def parse_table_rows(table, pdf_name):
    rows = smart_expand_rows(table)
    max_len = max(len(r) for r in rows if r) if rows else 0
    if max_len <= 3:
        return parse_compact_table(table, pdf_name)

    def get_cell(row, idx):
        return normalize_cell(row[idx]) if idx < len(row) else ""

    records = []
    for row in rows:
        tokens = row_tokens(row)
        if is_header_row(tokens):
            continue
        numeric_counts = [count_numeric_tokens(c) for c in row[2:]]
        numeric_region = " ".join(get_cell(row, i) for i in range(2, len(row)))
        if re.search(r"[A-Za-z]", numeric_region):
            token_record = parse_row_by_tokens(row, pdf_name)
            if token_record:
                records.append(token_record)
                continue
        if numeric_counts:
            total_nums = sum(numeric_counts)
            multi_cell = any(c >= 2 for c in numeric_counts)
            if multi_cell and total_nums >= 6 and len(row) < 10:
                token_record = parse_row_by_tokens(row, pdf_name)
                if token_record:
                    records.append(token_record)
                    continue
        crime_head = get_cell(row, 1)
        if not crime_head:
            continue
        if re.search(r"crime heads?", crime_head, flags=re.I):
            continue
        sr_text = get_cell(row, 0)
        sr_no = int(sr_text) if sr_text.isdigit() else None
        diff_sign = get_cell(row, 12)
        diff_val = get_cell(row, 13)

        records.append(
            {
                "pdf_file": pdf_name,
                "sr_no": sr_no,
                "crime_head": crime_head,
                "current_month_reg": get_cell(row, 2),
                "current_month_det": get_cell(row, 3),
                "previous_month_reg": get_cell(row, 4),
                "previous_month_det": get_cell(row, 5),
                "current_year_reg": get_cell(row, 6),
                "current_year_det": get_cell(row, 7),
                "current_year_det_pct": get_cell(row, 8),
                "previous_year_reg": get_cell(row, 9),
                "previous_year_det": get_cell(row, 10),
                "previous_year_det_pct": get_cell(row, 11),
                "diff_reg_sign": diff_sign if diff_sign in {"+", "-"} else diff_sign,
                "diff_reg_value": diff_val,
            }
        )
    return records


def parse_page(page, pdf_name, page_index=0):
    pixel_lines = {}
    gap_lines = []
    header_gap_lines = []
    if USE_PIXEL_LINES:
        pixel_lines = detect_table_lines_from_image(page, pdf_name)
    if pixel_lines.get("vertical"):
        debug_suffix = "" if page_index == 0 else f"_page{page_index + 1}"
        gap_debug_prefix = None
        if EXPORT_GAP_DEBUG:
            gap_debug_prefix = DEBUG_DIR / f"{Path(pdf_name).stem}{debug_suffix}_gap"
        gap_header_debug_prefix = None
        if EXPORT_GAP_DEBUG_HEADER:
            gap_header_debug_prefix = DEBUG_DIR / f"{Path(pdf_name).stem}{debug_suffix}_gap_header"
        gap_lines, header_gap_lines = compute_whitespace_vertical_lines_from_image(
            page,
            pixel_lines=pixel_lines,
            debug_prefix=gap_debug_prefix,
            debug_header_prefix=gap_header_debug_prefix,
        )
        if gap_lines:
            merged = pixel_lines["vertical"] + gap_lines
            pixel_lines["vertical"] = sorted(cluster_positions(merged, tolerance=1))
    grid_debug = {}
    if EXPORT_GRID_DEBUG:
        grid_debug = extend_grid_debug_image(
            page,
            pdf_name,
            pixel_lines=pixel_lines,
            gap_lines=gap_lines,
            header_gap_lines=header_gap_lines,
            page_index=page_index,
        )
    if EXPORT_PAGE_DEBUG:
        export_page_debug(
            page,
            pdf_name,
            pixel_lines=pixel_lines,
            grid_debug=grid_debug,
            page_index=page_index,
        )
    if pixel_lines.get("vertical") and pixel_lines.get("horizontal"):
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
            records = parse_table_rows(table, pdf_name)
            if records:
                header_tags = build_header_date_tags(page, pixel_lines)
                for record in records:
                    record.update(header_tags)
            return records
    return []


def parse_pdf_pages(pdf_path, page_indices=None, group_by_page=False):
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return [] if not group_by_page else []
        if page_indices is None:
            page_indices = range(len(pdf.pages))
        records = []
        grouped = []
        pdf_name = Path(pdf_path).name
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(pdf.pages):
                continue
            page = pdf.pages[page_index]
            page_records = parse_page(page, pdf_name, page_index=page_index)
            if group_by_page:
                if page_records:
                    grouped.append({"page_index": page_index, "records": page_records})
            else:
                records.extend(page_records)
        return grouped if group_by_page else records


def parse_all_pdfs(pdf_dir, output_dir, all_pages=True, page_index=0):
    pdf_dir = Path(pdf_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return
    for pdf_path in pdf_paths:
        if all_pages:
            records_by_page = parse_pdf_pages(pdf_path, group_by_page=True)
            records = [row for page in records_by_page for row in page["records"]]
        else:
            records_by_page = []
            records = parse_pdf_pages(pdf_path, page_indices=[page_index])
        if not records:
            scope = "any pages" if all_pages else f"page {page_index}"
            print(f"No rows parsed from {pdf_path} ({scope}).")
            continue
        csv_path = out_dir / f"{pdf_path.stem}.csv"
        json_path = out_dir / f"{pdf_path.stem}.json"
        if all_pages:
            write_outputs(records, csv_path=csv_path, json_path=None)
            write_outputs(records_by_page, csv_path=None, json_path=json_path)
        else:
            write_outputs(records, csv_path=csv_path, json_path=json_path)
        print(f"Parsed {len(records)} rows from {pdf_path}")


def parse_first_page(pdf_path):
    return parse_pdf_pages(pdf_path, page_indices=[0])


def export_page_debug(page, pdf_name, pixel_lines=None, grid_debug=None, page_index=0):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "pdf_file": pdf_name,
        "page_index": page_index,
        "width": page.width,
        "height": page.height,
        "text": page.extract_text() or "",
        "tables": page.extract_tables() or [],
        "words": page.extract_words() or [],
        "pixel_lines": pixel_lines or {},
        "grid_debug": grid_debug or {},
    }
    out_path = DEBUG_DIR / f"{Path(pdf_name).stem}_page{page_index + 1}.json"
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


def main(
    pdf_path=None,
    csv_path="parsed_first_page.csv",
    json_path="parsed_first_page.json",
    all_pages=False,
    page_index=0,
    all_pdfs=False,
    pdf_dir="crime_pdfs",
    out_dir="parsed_outputs",
):
    page_index_provided = page_index is not None
    if page_index is None:
        page_index = 0
    if all_pdfs:
        if not page_index_provided and not all_pages:
            all_pages = True
        parse_all_pdfs(pdf_dir, out_dir, all_pages=all_pages, page_index=page_index)
        return
    if pdf_path is None:
        pdf_dir_path = Path(pdf_dir)
        candidates = sorted(pdf_dir_path.glob("*.pdf")) if pdf_dir_path.exists() else []
        if not candidates:
            print("No PDF provided and none found in crime_pdfs/.")
            return
        pdf_path = candidates[0]

    if all_pages:
        records_by_page = parse_pdf_pages(pdf_path, group_by_page=True)
        records = [row for page in records_by_page for row in page["records"]]
    else:
        records = parse_pdf_pages(pdf_path, page_indices=[page_index])
    if not records:
        scope = "any pages" if all_pages else f"page {page_index}"
        print(f"No rows parsed from {scope}.")
        return
    if all_pages:
        write_outputs(records, csv_path=csv_path, json_path=None)
        write_outputs(records_by_page, csv_path=None, json_path=json_path)
    else:
        write_outputs(records, csv_path=csv_path, json_path=json_path)
    print(f"Parsed {len(records)} rows from {pdf_path}")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse table rows from PDF pages.")
    parser.add_argument("pdf_path", nargs="?", default=None)
    parser.add_argument("--all-pages", action="store_true", help="Parse all pages")
    parser.add_argument("--page", type=int, default=None, help="0-based page index")
    parser.add_argument("--all-pdfs", action="store_true", help="Parse all PDFs in a directory")
    parser.add_argument("--pdf-dir", default="crime_pdfs", help="Directory containing PDFs")
    parser.add_argument("--out-dir", default="parsed_outputs", help="Output directory for --all-pdfs")
    parser.add_argument("--csv", dest="csv_path", default="parsed_first_page.csv")
    parser.add_argument("--json", dest="json_path", default="parsed_first_page.json")
    args = parser.parse_args()

    main(
        args.pdf_path,
        csv_path=args.csv_path,
        json_path=args.json_path,
        all_pages=args.all_pages,
        page_index=args.page,
        all_pdfs=args.all_pdfs,
        pdf_dir=args.pdf_dir,
        out_dir=args.out_dir,
    )
