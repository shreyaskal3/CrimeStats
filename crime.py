import argparse
import csv
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

URL = "https://mumbaipolice.gov.in/CrimeStatistics"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": URL,
}

parser = argparse.ArgumentParser(description="Download Mumbai Police crime statistics PDFs.")
parser.add_argument("--out", default="crime_pdfs", help="Output directory for PDFs and index.csv")
parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
args = parser.parse_args()

session = requests.Session()
session.headers.update(HEADERS)

resp = session.get(URL, timeout=30)
resp.raise_for_status()
html = resp.text


def extract_pdf_links(html_text, base_url):
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        hrefs = [a.get("href") for a in soup.select("a[href]")]
    except Exception:
        # Fallback if BeautifulSoup is not available
        hrefs = re.findall(r'href=["\'](.*?)["\']', html_text, flags=re.I)

    links = [urljoin(base_url, h) for h in hrefs if h]
    pdf_links = [l for l in links if l.lower().endswith(".pdf")]

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for link in pdf_links:
        if link not in seen:
            unique.append(link)
            seen.add(link)
    return unique


pdf_links = extract_pdf_links(html, URL)

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

def download_pdf(url, dest_dir, force=False):
    filename = url.split("/")[-1]
    path = dest_dir / filename
    if path.exists() and not force:
        return path, "skipped"
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path, "downloaded"


results = []
for link in pdf_links:
    path, status = download_pdf(link, out_dir, force=args.force)
    results.append((link, status, path))


downloaded = sum(1 for _, s, _ in results if s == "downloaded")
skipped = sum(1 for _, s, _ in results if s == "skipped")

rows = [
    {"file": p.name, "bytes": p.stat().st_size, "status": s, "url": u}
    for u, s, p in results
]

index_path = out_dir / "index.csv"
with open(index_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "bytes", "status", "url"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Found {len(pdf_links)} PDFs. Downloaded {downloaded}, skipped {skipped}.")
print(f"Saved index: {index_path}")
