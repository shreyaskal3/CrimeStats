import argparse
from pathlib import Path

import cv2
import numpy as np
import pdfplumber

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional local dependency
    plt = None


def load_page_gray(page, resolution=200):
    page_image = page.to_image(resolution=resolution)
    img = np.array(page_image.original)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return gray


def save_pixel_intensity_plot(gray, out_dir, stem):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it to run this script.")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stem}_pixel_intensity.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(gray, cmap="gray", origin="upper", vmin=0, vmax=255)
    plt.title("Pixel intensity (gray)")
    plt.xlabel("Column index (px)")
    plt.ylabel("Row index (px)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot row/column pixel projections for a PDF page.")
    parser.add_argument("pdf_path", nargs="?", default="crime_pdfs/1.pdf")
    parser.add_argument("--page", type=int, default=0, help="0-based page index")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--out-dir", default="page_debug")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages or args.page >= len(pdf.pages):
            raise ValueError(f"Page index {args.page} out of range for {pdf_path}")
        page = pdf.pages[args.page]
        gray = load_page_gray(page, resolution=args.resolution)

    out_dir = Path(args.out_dir)
    out_path = save_pixel_intensity_plot(gray, out_dir, pdf_path.stem)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
