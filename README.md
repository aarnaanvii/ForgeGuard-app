# ForgeGuard

A web app for detecting digitally tampered images, built on a CNN + SVM pipeline trained on the CASIA2 image forensics dataset.

## What it does

Upload an image, and ForgeGuard classifies it as **Authentic** or **Tampered**, along with a confidence percentage.

## How it works

1. **Patch extraction** — the image is split into 128×128 patches (small images are upscaled first to meet the minimum size).
2. **CNN feature extraction** — each patch passes through a CNN whose first convolutional layer uses fixed SRM (Spatial Rich Model) high-pass filters. These filters suppress visual content and amplify local noise residuals, since tampered regions often carry a different noise fingerprint than their surroundings (e.g. from a different camera sensor or compression pass). Later layers extract higher-level patterns from this noise representation. Patch outputs are mean-pooled into a single 400-dimensional feature vector.
3. **SVM classification** — the feature vector is passed to a pretrained SVM (trained on CASIA2), which outputs the Authentic/Tampered label. Since the SVM was trained without probability calibration, its raw decision score is passed through a sigmoid to produce an approximate confidence percentage.

## Tech stack

- **Backend:** Flask
- **ML:** PyTorch (CNN), scikit-learn (SVM), OpenCV, scikit-image
- **Frontend:** Simple HTML upload UI
- **Deployment:** Gunicorn (see `Procfile`)

## Running locally

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Project structure

```
app.py                  # Flask app: routes, model loading, inference
model/
  cnn.py                 # CNN architecture with fixed SRM filters
  patch_extraction.py     # Patch extraction + feature vector generation
templates/
  index.html              # Upload UI
weights/
  cnn_casia2.pt            # Pretrained CNN weights
  svm_casia2.pt            # Pretrained SVM classifier
requirements.txt
Procfile                 # Gunicorn config for deployment
```

## Dataset

Trained on [CASIA2](https://github.com/namtpham/casia2groundtruth), a widely used benchmark dataset for image forgery detection research.
