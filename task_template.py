import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
ZIP_FILE = "Dataset.zip"                 # Path to the downloaded dataset zip
DATASET_DIR = Path("Dataset")             # Unzipped folder
TEMP_OUT_DIR = Path("submission_temp")    # Temporary folder for forged images
FILE_PATH = "submission.zip"              # Final file to upload

CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

# Groups that carry a genuinely recoverable, content-independent watermark
# pattern (confirmed via split-half reliability testing: correlation
# 0.28-0.65 across independent halves of the source set, vs <0.03 for the
# remaining groups under the same test). These use template averaging.
TEMPLATE_SETTINGS = {
    "WM_4": dict(sigma=3.0, band=(0.5, 1.0), alpha=1.67),
    "WM_5": dict(sigma=0.5, band=(0.5, 1.0), alpha=7.03),
    "WM_6": dict(sigma=0.5, band=(0.5, 1.0), alpha=3.22),
}

# Groups that resisted averaging entirely (tested across pixel, DCT, and
# wavelet domains, JND-normalized averaging, and every blend size from
# single-copy to near-full-averaging -- all near-zero split-half
# correlation). These use a watermark copy attack: each target is paired
# with its k closest content-matched sources and their residuals are
# blended and transplanted directly, preserving content-adaptive
# structure that averaging would otherwise destroy.
KNN_ALPHA = {
    "WM_1": 0.759, "WM_2": 0.660, "WM_3": 0.725, "WM_7": 0.462, "WM_8": 0.373,
}
KNN_K = 3

# Alpha values for both methods were tuned by sweeping embedding strength
# and measuring LPIPS locally, then (for WM_4/5/6) confirmed against real
# leaderboard scores via a systematic bracket search (40%-100% of an
# initial estimate), which revealed an interior optimum at 60% rather than
# a monotonic "more signal = better detection" relationship.


# ============================================================
# 1. UNZIP DATASET
# ============================================================
if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Shared helpers
# ============================================================
def to_array(img):
    return np.asarray(img, dtype=np.float32)

def high_freq_residual(img_arr, blur_sigma):
    blurred = cv2.GaussianBlur(img_arr, ksize=(0, 0), sigmaX=blur_sigma)
    return img_arr - blurred

def bandpass_filter(residual, low_frac, high_frac):
    h, w, c = residual.shape
    out = np.zeros_like(residual)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    mask = (dist >= low_frac * max_dist) & (dist <= high_frac * max_dist)
    for ch in range(c):
        f = np.fft.fftshift(np.fft.fft2(residual[:, :, ch]))
        out[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    return out

def stack_to_common_shape(arrs):
    shapes = [a.shape for a in arrs]
    target = max(set(shapes), key=shapes.count)
    return [a if a.shape == target else cv2.resize(a, (target[1], target[0])) for a in arrs]

def apply_template(clean_arr, template, alpha):
    h, w = clean_arr.shape[:2]
    t = template if template.shape[:2] == (h, w) else cv2.resize(template, (w, h))
    forged = np.clip(clean_arr + alpha * t, 0, 255).astype(np.uint8)
    return forged

def small_gray(pil_img, size=32):
    return np.asarray(pil_img.convert("L").resize((size, size)), dtype=np.float32).flatten()


# ============================================================
# 2. FORGERY ATTACK
# ============================================================
print("Building forgery submission...")

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} dataset -> Forging onto images {target_start}.png to {target_stop}.png ...")

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(source_dir.glob("*.png"))
    if not source_paths:
        print(f"  [Warning] No source images found in {source_dir}")
        continue

    target_dir = DATASET_DIR / "clean_targets"
    target_paths = [target_dir / f"{n}.png" for n in range(target_start, target_stop + 1)]

    source_images = [Image.open(p).convert("RGB") for p in source_paths]

    if source_wm in TEMPLATE_SETTINGS:
        # --- Template averaging: estimate one shared watermark pattern
        # from all 25 source images, apply it to every target in this batch ---
        cfg = TEMPLATE_SETTINGS[source_wm]
        residuals = [high_freq_residual(to_array(img), cfg["sigma"]) for img in source_images]
        avg_residual = np.mean(stack_to_common_shape(residuals), axis=0)
        template = bandpass_filter(avg_residual, *cfg["band"])

        for target_path in target_paths:
            target_pil = Image.open(target_path).convert("RGB")
            target_arr = to_array(target_pil)
            forged_img = apply_template(target_arr, template, cfg["alpha"])
            Image.fromarray(forged_img).save(TEMP_OUT_DIR / target_path.name)
            total_processed += 1

    else:
        # --- Watermark copy attack (k-NN blend): for each target, blend
        # the k most visually-similar source residuals and transplant ---
        alpha = KNN_ALPHA[source_wm]
        src_vecs = [small_gray(img) for img in source_images]

        for target_path in target_paths:
            target_pil = Image.open(target_path).convert("RGB")
            t_vec = small_gray(target_pil)

            dists = [np.sum((t_vec - s_vec) ** 2) for s_vec in src_vecs]
            nearest_k = np.argsort(dists)[:KNN_K]

            residuals = []
            for idx in nearest_k:
                r = high_freq_residual(to_array(source_images[idx]), 1.5)
                r = bandpass_filter(r, 0.05, 0.45)
                residuals.append(r)
            blended_residual = np.mean(residuals, axis=0)

            target_arr = to_array(target_pil)
            forged_img = apply_template(target_arr, blended_residual, alpha)
            Image.fromarray(forged_img).save(TEMP_OUT_DIR / target_path.name)
            total_processed += 1

print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200 images, but processed {total_processed}. Your submission may be rejected!")


# ============================================================
# 3. PACKAGE INTO FLAT ZIP FILE
# ============================================================
print(f"Packaging images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)
print(f"Saved submission file to {FILE_PATH}")
