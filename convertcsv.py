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


def detect_header_block(df: pd.DataFrame):
    """
    Detect multi-row headers where one row contains R/D metrics and earlier rows contain dates.
    Returns (date_rows_idx, metric_row_idx) or None.
    """
    max_scan = min(6, len(df))
    for i in range(max_scan):
        row = df.iloc[i].astype(str)
        has_rd = any(v.strip().upper() in {"R", "D"} for v in row)
        if not has_rd:
            continue
        has_date = False
        for r in range(i):
            for v in df.iloc[r].astype(str):
                if re.search(r"\d{2}\.\d{2}\.\d{4}", v):
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
        if col_str and not col_str.lower().startswith("unnamed"):
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


# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# If the table encodes the real headers in the first two rows (date row + R/D row),
# normalize them into standard column names like "31.01.2018_R".
period_labels_by_idx = None
header_block = detect_header_block(df)
if header_block:
    date_rows_idx, metric_row_idx = header_block
    period_labels_by_idx = build_period_labels_from_rows(df, date_rows_idx)
    new_cols = []
    last_date = None
    for col_idx, col in enumerate(df.columns):
        # scan date rows for this column
        for r in date_rows_idx:
            cell = str(df.iat[r, col_idx])
            m = re.search(r"(\d{2}\.\d{2}\.\d{4})", cell)
            if m:
                last_date = m.group(1)
                break
        date_str = last_date
        metric = str(df.iat[metric_row_idx, col_idx]).strip().upper()
        if metric in {"R", "D"} and date_str:
            new_cols.append(f"{date_str}_{metric}")
        else:
            new_cols.append(col)
    df = df.iloc[metric_row_idx + 1 :].copy()
    df.columns = new_cols
    df.reset_index(drop=True, inplace=True)

# ---- Heuristics for id columns ----
# Try to locate crime name + code columns.
# Adjust these if your CSV uses different labels.
possible_code_cols = [c for c in df.columns if c.strip().lower() in {"sr", "sno", "no", "code", "crime_code"}]
possible_name_cols = [c for c in df.columns if "crime" in c.strip().lower() or c.strip().lower() in {"head", "category", "name"}]

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
