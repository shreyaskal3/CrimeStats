import os
import re
import pandas as pd
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = os.environ.get("INPUT_CSV", "extracted_tables/2/table_1_page_1.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "crime_monthly_stats_2p1.csv")

# Optional: if your CSV has extra junk rows like Total, put keywords here
DROP_NAME_KEYWORDS = {"total"}
# If the first two rows contain multi-line headers (date + R/D),
# keep only period columns that include these keywords (case-insensitive).
# Set to None to keep all periods (Month/Year/etc.).
KEEP_PERIOD_KEYWORDS = {"month"}

# -----------------------------
# Helpers
# -----------------------------
def parse_int(x):
    """
    Convert values like '1,484' -> 1484, ' - 3' -> -3, '+ 4' -> 4, '' -> NA.
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    if s == "" or s.lower() in {"-", "—", "na", "n/a", "none"}:
        return pd.NA

    # keep digits and optional sign; remove commas/spaces
    s = s.replace(",", "")
    # handle cases like '+ 4', '- 3'
    s = re.sub(r"\s+", "", s)

    # if it's something like "492+" or "492 +" -> 492
    s = re.sub(r"\+$", "", s)

    # extract signed integer if present
    m = re.match(r"^[+-]?\d+$", s)
    if m:
        return int(s)

    # fallback: try extracting first signed int inside string
    m = re.search(r"[+-]?\d+", s)
    if m:
        return int(m.group(0))

    return pd.NA


def normalize_date(colname: str):
    """
    Detect dd.mm.yyyy in a column name and return YYYY-MM-DD.
    Examples:
      '31.01.2018_R' -> '2018-01-31'
      'R_31.01.2018' -> '2018-01-31'
    """
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})", colname)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%d.%m.%Y").date()
    return dt.isoformat()


def extract_date_from_header(text, next_text=None, last_year=None):
    """
    Extract the last date from a header cell. Supports dd.mm.yyyy or dd.mm. + year.
    Returns (date_str, year_str).
    """
    s = str(text)
    candidates = []
    for m in re.finditer(r"\d{2}\.\d{2}\.\d{4}", s):
        candidates.append((m.start(), "full", m.group(0)))
    for m in re.finditer(r"\d{2}\.\d{2}\.(?!\d{4})", s):
        candidates.append((m.start(), "short", m.group(0)))
    if not candidates:
        return None, last_year
    _, kind, val = max(candidates, key=lambda x: x[0])
    if kind == "full":
        return val, val[-4:]

    year = None
    if next_text is not None:
        m = re.search(r"\b(\d{4})\b", str(next_text))
        if m:
            year = m.group(1)
    if not year:
        years = re.findall(r"\b(\d{4})\b", s)
        if years:
            year = years[-1]
    if not year:
        year = last_year
    if not year:
        return None, last_year

    if val.endswith("."):
        val = val[:-1]
    return f"{val}.{year}", year


def metric_from_col(colname: str):
    """
    Decide if column is R or D.
    Works for headers containing standalone R/D or _R/_D.
    """
    # common patterns
    if re.search(r"(^|[^A-Z])R([^A-Z]|$)", colname.upper()):
        # but avoid picking up "RAPE" etc (rare in headers)
        # Prefer explicit separators:
        if re.search(r"(_R$|R_| R |\(R\))", colname.upper()):
            return "R"
    if re.search(r"(^|[^A-Z])D([^A-Z]|$)", colname.upper()):
        if re.search(r"(_D$|D_| D |\(D\))", colname.upper()):
            return "D"

    # simpler fallbacks
    if colname.strip().upper().endswith(("_R", " R")):
        return "R"
    if colname.strip().upper().endswith(("_D", " D")):
        return "D"

    return None


def metric_token_count(row) -> int:
    count = 0
    for v in row:
        t = str(v).strip().upper()
        if t in {"R", "D", "REG.", "DET.", "REG", "DET"}:
            count += 1
            continue
        if re.search(r"\bR\s*[/&]?\s*D\b", t) or t == "RD":
            count += 2
    return count


def detect_header_block(df: pd.DataFrame):
    """
    Detect multi-row headers where one row contains R/D metrics and earlier rows contain dates.
    Returns (date_rows_idx, metric_row_idx) or None.
    """
    max_scan = min(6, len(df))
    for i in range(max_scan):
        row = df.iloc[i].astype(str)
        if metric_token_count(row) < 2:
            continue
        has_date = False
        for r in range(i):
            for v in df.iloc[r].astype(str):
                if re.search(r"\d{2}\.\d{2}\.(\d{4})?", v):
                    has_date = True
                    break
            if has_date:
                break
        if has_date:
            return list(range(i)), i
    return None


def build_period_labels(columns):
    """
    Build a period label per column based on the nearest non-unnamed header.
    This helps filter Month vs Year groups when headers are multi-line.
    """
    labels = []
    last_label = None
    for col in columns:
        col_str = str(col)
        is_year_only = re.fullmatch(r"\d{4}(\.\d+)?", col_str) is not None
        if (
            col_str
            and col_str.lower() != "nan"
            and not col_str.lower().startswith("unnamed")
            and not is_year_only
        ):
            last_label = col_str
        labels.append(last_label or col_str)
    return labels


def build_period_labels_from_rows(df: pd.DataFrame, rows: list):
    labels = []
    for col_idx in range(df.shape[1]):
        parts = []
        for r in rows:
            if 0 <= r < len(df):
                val = str(df.iat[r, col_idx]).strip()
                if val and val.lower() != "nan":
                    parts.append(val)
        labels.append(" ".join(parts).strip())

    # Propagate labels across adjacent columns (e.g., when year is split into the next column).
    filled = []
    last = ""
    for label in labels:
        if re.fullmatch(r"\d{4}", label):
            label = ""
        if label:
            last = label
            filled.append(label)
        else:
            filled.append(last)
    return filled


# Return True if a row looks like a metric row (R/D markers).
def row_is_metric(row) -> bool:
    return metric_token_count(row) >= 2


def columns_have_dates(columns) -> bool:
    return any(re.search(r"\d{2}\.\d{2}\.(\d{4})?", str(c)) for c in columns)


def is_combined_metric(val: str) -> bool:
    s = str(val).strip().upper()
    return bool(re.search(r"\bR\s*[/&]?\s*D\b", s))


def normalize_metric_token(val: str):
    s = str(val).strip().upper()
    if s in {"R", "REG.", "REG"}:
        return "R"
    if s in {"D", "DET.", "DET"}:
        return "D"
    return s


def split_rd_pair(val):
    if pd.isna(val):
        return (pd.NA, pd.NA)
    s = str(val).strip()
    if s == "" or s.lower() in {"-", "—", "na", "n/a", "none"}:
        return (pd.NA, pd.NA)
    nums = re.findall(r"[+-]?\d[\d,]*", s)
    if len(nums) >= 2:
        return (parse_int(nums[0]), parse_int(nums[1]))
    if len(nums) == 1:
        return (parse_int(nums[0]), pd.NA)
    return (pd.NA, pd.NA)


def expand_combined_rd_columns(df: pd.DataFrame, period_labels_by_idx):
    if not any(str(c).endswith("_RD") for c in df.columns):
        return df, period_labels_by_idx
    def _unique_name(base, existing, new_cols):
        name = base
        i = 1
        while name in existing or name in new_cols:
            name = f"{base}_{i}"
            i += 1
        return name

    new_series = []
    new_cols = []
    new_labels = [] if period_labels_by_idx is not None else None
    for idx, col in enumerate(df.columns):
        label = period_labels_by_idx[idx] if period_labels_by_idx is not None else None
        if str(col).endswith("_RD"):
            date_str = str(col)[:-3]
            r_col = f"{date_str}_R"
            d_col = f"{date_str}_D"
            if r_col in df.columns or d_col in df.columns or r_col in new_cols or d_col in new_cols:
                r_col = _unique_name(f"{date_str}_RD_R", df.columns, new_cols)
                d_col = _unique_name(f"{date_str}_RD_D", df.columns, new_cols)
            r_vals = []
            d_vals = []
            for v in df.iloc[:, idx].tolist():
                r, d = split_rd_pair(v)
                r_vals.append(r)
                d_vals.append(d)
            new_series.append(pd.Series(r_vals, index=df.index))
            new_cols.append(r_col)
            new_series.append(pd.Series(d_vals, index=df.index))
            new_cols.append(d_col)
            if new_labels is not None:
                new_labels.extend([label, label])
        else:
            new_series.append(df.iloc[:, idx])
            new_cols.append(col)
            if new_labels is not None:
                new_labels.append(label)
    df = pd.concat(new_series, axis=1)
    df.columns = new_cols
    return df, new_labels


def align_period_labels(period_labels_by_idx, columns):
    if period_labels_by_idx is None:
        return None
    if len(period_labels_by_idx) == len(columns):
        return period_labels_by_idx
    # If labels are shorter, extend by repeating last known label.
    aligned = list(period_labels_by_idx)
    last = aligned[-1] if aligned else ""
    while len(aligned) < len(columns):
        aligned.append(last)
    # If labels are longer, truncate.
    return aligned[: len(columns)]


# If the CSV has a title row and real headers are on the first data row,
# promote that row to headers (e.g., "Sr. No.", "Crime Heads").
def maybe_promote_header_row(df: pd.DataFrame) -> pd.DataFrame:
    header_markers = [r"\bsr\.?\s*no\b", r"\bcrime heads?\b"]
    if any(re.search(pat, str(c), re.IGNORECASE) for pat in header_markers for c in df.columns):
        return df
    max_scan = min(3, len(df))
    for i in range(max_scan):
        row_vals = df.iloc[i].astype(str).tolist()
        if any(re.search(r"\bsr\.?\s*no\b", v, re.IGNORECASE) for v in row_vals) and any(
            re.search(r"\bcrime heads?\b", v, re.IGNORECASE) for v in row_vals
        ):
            df2 = df.iloc[i + 1 :].copy()
            df2.columns = row_vals
            df2.reset_index(drop=True, inplace=True)
            return df2
    return df


def expand_serial_numbers(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    Some PDFs pack multiple serial numbers into one cell (e.g., "1 2 3 4 5").
    Expand those into subsequent empty rows.
    """
    if code_col in df.columns:
        df[code_col] = df[code_col].astype("object")
    for i in range(len(df)):
        raw = df.at[i, code_col] if code_col in df.columns else None
        if pd.isna(raw):
            continue
        nums = re.findall(r"\d+", str(raw))
        if len(nums) <= 1:
            continue
        # Assign first number to current row
        df.at[i, code_col] = nums[0]
        # Fill down into subsequent empty rows
        for j, num in enumerate(nums[1:], start=1):
            if i + j >= len(df):
                break
            nxt = df.at[i + j, code_col]
            if pd.isna(nxt) or str(nxt).strip() == "":
                df.at[i + j, code_col] = num
            else:
                break
    return df


# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(INPUT_CSV)
df = maybe_promote_header_row(df)

# If the table encodes the real headers in the first two rows (date row + R/D row),
# normalize them into standard column names like "31.01.2018_R".
period_labels_by_idx = None
header_block = detect_header_block(df)
if not header_block and columns_have_dates(df.columns):
    max_scan = min(3, len(df))
    for i in range(max_scan):
        if row_is_metric(df.iloc[i].astype(str)):
            header_block = ([], i)
            break
if header_block:
    date_rows_idx, metric_row_idx = header_block
    # Prefer column headers when they contain the keywords we actually filter on;
    # otherwise fall back to multi-row labels (common when headers are split).
    def _labels_contain_keywords(labels, keywords):
        if not keywords:
            return False
        for lbl in labels:
            if any(k in (lbl or "").lower() for k in keywords):
                return True
        return False

    col_labels = build_period_labels(df.columns)
    row_labels = build_period_labels_from_rows(df, date_rows_idx) if date_rows_idx else []
    if KEEP_PERIOD_KEYWORDS:
        if _labels_contain_keywords(col_labels, KEEP_PERIOD_KEYWORDS):
            period_labels_by_idx = col_labels
        elif _labels_contain_keywords(row_labels, KEEP_PERIOD_KEYWORDS):
            period_labels_by_idx = row_labels
        else:
            period_labels_by_idx = col_labels
    else:
        period_labels_by_idx = col_labels if any(
            re.search(r"\b(month|year)\b", str(c), re.IGNORECASE) for c in df.columns
        ) else row_labels
    new_cols = []
    last_date = None
    last_year = None
    for col_idx, col in enumerate(df.columns):
        # scan date rows for this column
        found_date = None
        for r in date_rows_idx:
            cell = str(df.iat[r, col_idx])
            m = re.search(r"(\d{2}\.\d{2}\.\d{4})", cell)
            if m:
                found_date = m.group(1)
        if found_date:
            last_date = found_date
            last_year = found_date[-4:]
            date_str = found_date
        else:
            date_str = None
            # If date rows are present but blank for this column, try header text.
            next_col = df.columns[col_idx + 1] if col_idx + 1 < len(df.columns) else None
            extracted, last_year = extract_date_from_header(col, next_col, last_year)
            if extracted:
                date_str = extracted
                last_date = extracted
            else:
                date_str = last_date
        if not date_rows_idx:
            # Prefer a date from the previous row (often the date row),
            # otherwise try extracting from the column header itself.
            prev_row = (
                df.iat[metric_row_idx - 1, col_idx] if metric_row_idx - 1 >= 0 else None
            )
            prev_next = (
                df.iat[metric_row_idx - 1, col_idx + 1]
                if metric_row_idx - 1 >= 0 and col_idx + 1 < len(df.columns)
                else None
            )
            extracted, last_year = extract_date_from_header(prev_row, prev_next, last_year)
            if not extracted:
                next_col = df.columns[col_idx + 1] if col_idx + 1 < len(df.columns) else None
                extracted, last_year = extract_date_from_header(col, next_col, last_year)
            if extracted:
                date_str = extracted
                last_date = extracted
        metric = normalize_metric_token(df.iat[metric_row_idx, col_idx])
        if date_str and is_combined_metric(metric):
            new_cols.append(f"{date_str}_RD")
        elif metric in {"R", "D"} and date_str:
            new_cols.append(f"{date_str}_{metric}")
        else:
            new_cols.append(col)
    df = df.iloc[metric_row_idx + 1 :].copy()
    df.columns = new_cols
    df.reset_index(drop=True, inplace=True)
    df, period_labels_by_idx = expand_combined_rd_columns(df, period_labels_by_idx)
    period_labels_by_idx = align_period_labels(period_labels_by_idx, df.columns)

# ---- Heuristics for id columns ----
# Try to locate crime name + code columns.
# Adjust these if your CSV uses different labels.
possible_code_cols = []
possible_name_cols = []
for c in df.columns:
    lc = str(c).strip().lower()
    if lc in {"sr", "sno", "no", "code", "crime_code", "sr. no."} or (
        re.search(r"\bsr\.?\b", lc) and re.search(r"\bno\b", lc)
    ):
        possible_code_cols.append(c)
    if ("crime" in lc and "sr" not in lc) or lc in {"head", "category", "name"}:
        possible_name_cols.append(c)

if not possible_code_cols:
    # Often first column is serial number
    possible_code_cols = [df.columns[0]]
if not possible_name_cols:
    # Often second column is crime head/name
    if len(df.columns) >= 2:
        possible_name_cols = [df.columns[1]]
    else:
        raise ValueError("Could not infer crime name column. Please rename your columns or edit the script.")

code_col = possible_code_cols[0]
name_col = possible_name_cols[0]
if name_col == code_col:
    # Pick a different column (often an Unnamed column holding crime names)
    for c in df.columns:
        if c != code_col:
            name_col = c
            break
    if name_col == code_col:
        raise ValueError(
            "Could not infer a distinct crime name column. "
            "Please rename your columns or edit the script."
        )

# Expand multi-serial cells before parsing to int
df = expand_serial_numbers(df, code_col)

# Clean crime_code and crime_name
df[code_col] = df[code_col].apply(parse_int)
df[name_col] = df[name_col].astype(str).str.strip()

# Drop total-like rows
df = df[~df[name_col].str.lower().isin(DROP_NAME_KEYWORDS)].copy()

# -----------------------------
# Identify date+metric columns
# -----------------------------
value_cols = []
col_meta = []  # (col, date, metric)

for idx, c in enumerate(df.columns):
    if c in {code_col, name_col}:
        continue

    if period_labels_by_idx and KEEP_PERIOD_KEYWORDS:
        label = period_labels_by_idx[idx] or ""
        if not any(k in label.lower() for k in KEEP_PERIOD_KEYWORDS):
            continue

    dt = normalize_date(c)
    met = metric_from_col(c)

    # Sometimes headers are two-level flattened poorly; if metric isn't in header,
    # we still accept date columns and handle later if user has separate R/D columns.
    if dt and met:
        value_cols.append(c)
        col_meta.append((c, dt, met))

# If nothing matched, assume columns come in pairs like:
# 31.01.2018_R, 31.01.2018_D etc — or even repeated date columns.
if not col_meta:
    # Try harder: look for date columns first, then infer R/D by alternating order
    date_cols = []
    for idx, c in enumerate(df.columns):
        if c in {code_col, name_col}:
            continue
        if period_labels_by_idx and KEEP_PERIOD_KEYWORDS:
            label = period_labels_by_idx[idx] or ""
            if not any(k in label.lower() for k in KEEP_PERIOD_KEYWORDS):
                continue
        dt = normalize_date(c)
        if dt:
            date_cols.append((c, dt))

    if not date_cols:
        raise ValueError(
            "Could not find any date columns in headers. "
            "Please ensure headers include dd.mm.yyyy (e.g., 31.01.2018_R)."
        )

    # Infer metric by order: first R then D for each date (common)
    # Group by date
    from collections import defaultdict
    grouped = defaultdict(list)
    for c, dt in date_cols:
        grouped[dt].append(c)

    for dt, cols in grouped.items():
        # sort for stability
        cols_sorted = cols[:]
        # If exactly 2 cols, map first->R second->D
        if len(cols_sorted) >= 2:
            col_meta.append((cols_sorted[0], dt, "R"))
            col_meta.append((cols_sorted[1], dt, "D"))
        else:
            # Can't infer with only one column
            raise ValueError(f"Only one column found for date {dt}. Need both R and D.")

    value_cols = [c for c, _, _ in col_meta]

# Guard against duplicate (date, metric) pairs which would collapse in the pivot.
if col_meta:
    from collections import defaultdict
    dup_map = defaultdict(list)
    for c, dt, met in col_meta:
        dup_map[(dt, met)].append(c)
    dups = {k: v for k, v in dup_map.items() if len(v) > 1}
    if dups:
        details = "; ".join(
            f"{dt} {met}: {', '.join(cols)}" for (dt, met), cols in dups.items()
        )
        raise ValueError(
            "Multiple columns map to the same date+metric. "
            "This usually happens when Month and Year columns share the same end date. "
            "Set KEEP_PERIOD_KEYWORDS to filter, or rename columns to disambiguate. "
            f"Conflicts: {details}"
        )

# -----------------------------
# Melt to long format
# -----------------------------
long = df[[code_col, name_col] + value_cols].melt(
    id_vars=[code_col, name_col],
    value_vars=value_cols,
    var_name="source_col",
    value_name="value"
)

# Attach date + metric
meta_df = pd.DataFrame(col_meta, columns=["source_col", "month_end_date", "metric"])
long = long.merge(meta_df, on="source_col", how="left")

# Clean numeric values
long["value"] = long["value"].apply(parse_int)

# Pivot metric back to columns: registered/detected
out = long.pivot_table(
    index=[code_col, name_col, "month_end_date"],
    columns="metric",
    values="value",
    aggfunc="first"
).reset_index()

# Rename
out = out.rename(columns={
    code_col: "crime_code",
    name_col: "crime_name",
    "R": "registered",
    "D": "detected"
})

# Ensure integer columns (nullable)
for col in ["crime_code", "registered", "detected"]:
    if col in out.columns:
        out[col] = out[col].astype("Int64")

# Sort nicely
out = out.sort_values(["month_end_date", "crime_code"]).reset_index(drop=True)

# Save
out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved normalized data to: {OUTPUT_CSV}")
print(out.head(10))
