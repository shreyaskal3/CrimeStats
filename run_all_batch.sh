#!/usr/bin/env bash
set -euo pipefail

pdf_dir=${1:-crime_pdfs}
layout_dir=${2:-layout_outputs}
extracted_root=${3:-extracted_tables}
output_dir=${4:-somefolder}
input_csv_rel=${5:-table_1_page_1.csv}

mkdir -p "$layout_dir" "$extracted_root" "$output_dir"

shopt -s nullglob
for pdf_path in "$pdf_dir"/*.pdf "$pdf_dir"/*.PDF; do
  pdf_base=$(basename "$pdf_path")
  stem=${pdf_base%.*}

  layout_json="$layout_dir/$stem.json"
  out_dir="$extracted_root/$stem"
  input_csv="$out_dir/$input_csv_rel"
  output_csv="$output_dir/crime_monthly_stats_${stem}.csv"

  ./run_all.sh "$pdf_path" "$layout_json" "$out_dir" "$input_csv" "$output_csv"
  echo "Finished $pdf_base -> $output_csv"
  echo "---"
 done
