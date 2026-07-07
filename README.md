# Assignment_4_-_Watermark_Forging

## Approach summary

Two attack methods, applied per watermarking scheme based on diagnostic testing:

- **WM_4, WM_5, WM_6** — Template averaging. These three schemes carry a
  fixed, content-independent watermark pattern. Averaging high-frequency
  residuals across the 25 source images reinforces the shared pattern
  while natural image content cancels out. A band-pass filter isolates
  the specific frequency range where each scheme's signal concentrates.

- **WM_1, WM_2, WM_3, WM_7, WM_8** — Watermark copy attack (k-NN blend).
  These schemes resisted averaging in every domain tested (pixel, DCT,
  wavelet, JND-normalized, and every blend size from single-copy to
  near-full averaging), consistent with content-adaptive/learned
  watermarking. Each target image is instead paired with its 3 most
  visually-similar source images, and their residuals are blended and
  transplanted directly.

Embedding strength for every group was tuned via local LPIPS sweeps; for
WM_4/5/6, alpha was further refined against real leaderboard scores
through a systematic bracket search, which revealed an interior optimum
(stronger embedding does not monotonically improve detection).

## How to reproduce our best result

1. **Install dependencies:**
   ```bash
   pip install numpy opencv-python-headless Pillow requests --break-system-packages
   ```

2. **Download the dataset** from HuggingFace and place `Dataset.zip` in
   the same folder as `task_template.py`:
   ```
   https://huggingface.co/datasets/SprintML/tml2026_task4
   ```

3. **Run the attack:**
   ```bash
   python task_template.py
   ```
   This unzips the dataset (if not already unzipped), forges all 200
   images using the tuned settings described above, and packages them
   into `submission.zip`.

4. **Submit:**
   - Open `submission.py`
   - Replace `API_KEY` with your team's API key
   - Confirm `FILE_PATH` points to `submission.zip`
   - Run:
     ```bash
     python submission.py
     ```

## Files

- `task_template.py` — dataset setup, forging attack, and packaging
- `submission.py` — leaderboard submission

Note: the leaderboard accepts one submission per 60 minutes on success
(2 minutes on failure).
