"""
preprocess_cars.py
==================
Applies the 6 essential preprocessing steps to every car image and
produces two PDF reports:
  • before_preprocessing.pdf  – original images laid out by class
  • after_preprocessing.pdf   – preprocessed images laid out by class

Preprocessing pipeline (in order):
  1. Resize to 224×224
  2. CLAHE  (contrast enhancement in LAB space)
  3. Gaussian Blur  (noise reduction)
  4. Sharpen  (unsharp-mask to recover edge detail lost in blur)
  5. Normalize  (min-max → uint8, balances overall exposure)
  6. Horizontal Flip augmentation (reflects cars for invariance)

Usage:
    python preprocess_cars.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT  = "dataset/Images"
OUT_BEFORE = "outputs/before_preprocessing.pdf"
OUT_AFTER  = "outputs/after_preprocessing.pdf"
IMG_SIZE   = 224
SPLITS     = ["Train", "Test"]
CLASSES    = ["audi", "lamborghini", "mercedes"]
COLS       = 5          # images per row in the PDF grid


# ── Core preprocessing steps ─────────────────────────────────────────────────

def resize(img: np.ndarray) -> np.ndarray:
    """Step 1 – Resize to IMG_SIZE × IMG_SIZE."""
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def clahe(img: np.ndarray) -> np.ndarray:
    """Step 2 – CLAHE contrast enhancement (operates in LAB L-channel)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((eq, a, b)), cv2.COLOR_LAB2BGR)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Step 3 – Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Step 4 – Unsharp-mask sharpening to restore edge clarity."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)


def normalize(img: np.ndarray) -> np.ndarray:
    """Step 5 – Min-max normalise to [0, 255] (balances exposure)."""
    f = img.astype(np.float32)
    n = (f - f.min()) / (f.max() - f.min() + 1e-7)
    return (n * 255).astype(np.uint8)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    """Step 6 – Horizontal flip for augmentation."""
    return cv2.flip(img, 1)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Apply all 6 steps in sequence."""
    img = resize(img_bgr)
    img = clahe(img)
    img = gaussian_blur(img)
    img = sharpen(img)
    img = normalize(img)
    img = horizontal_flip(img)
    return img


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_images() -> list[dict]:
    """
    Walk DATA_ROOT and return a list of dicts:
      { path, split, cls, original (BGR), processed (BGR) }
    Sorted by split → class → filename.
    """
    records = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = Path(DATA_ROOT) / split / cls
            if not folder.exists():
                continue
            for fp in sorted(folder.glob("*.jpg")):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                records.append({
                    "path":      fp,
                    "split":     split,
                    "cls":       cls,
                    "original":  resize(img),          # resize only for "before"
                    "processed": preprocess(img),
                })
    return records


# ── PDF helpers ───────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_pdf(records: list[dict], key: str, pdf_path: str, title: str):
    """
    Save a multi-page PDF.
    Each page = one class (all splits together), images in a COLS-wide grid.
    key: 'original' or 'processed'
    """
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for cls in CLASSES:
            imgs = [r[key] for r in records if r["cls"] == cls]
            labels = [f"{r['split']}/{r['path'].stem}" for r in records if r["cls"] == cls]
            n    = len(imgs)
            rows = (n + COLS - 1) // COLS

            fig, axes = plt.subplots(rows, COLS,
                                     figsize=(COLS * 2.8, rows * 2.8 + 0.8),
                                     squeeze=False)
            fig.suptitle(f"{title}  —  {cls.capitalize()}  ({n} images)",
                         fontsize=14, fontweight="bold")

            for idx, ax in enumerate(axes.flat):
                if idx < n:
                    ax.imshow(bgr_to_rgb(imgs[idx]))
                    ax.set_title(labels[idx], fontsize=6, pad=2)
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
            print(f"  [{key}] {cls}: {n} images written")

    print(f"Saved → {pdf_path}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading all images …")
    records = load_all_images()
    print(f"Loaded {len(records)} images total\n")

    print("Building before_preprocessing.pdf …")
    save_pdf(records, "original",  OUT_BEFORE, "Before Preprocessing")

    print("Building after_preprocessing.pdf …")
    save_pdf(records, "processed", OUT_AFTER,  "After Preprocessing")

    print("Done. Files saved to outputs/")


if __name__ == "__main__":
    main()
