import io
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from joblib import load as joblib_load

from model.cnn import CNN
from model.patch_extraction import get_patch_yi

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "cnn_casia2.pt")
SVM_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "svm_casia2.pt")

_cnn_model = None
_svm_model = None


def load_models():
    """Load the pretrained CNN + SVM once, at process startup."""
    global _cnn_model, _svm_model

    with torch.no_grad():
        cnn = CNN()
        cnn.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=torch.device("cpu")))
        cnn.eval()
        cnn = cnn.double()
    _cnn_model = cnn

    _svm_model = joblib_load(SVM_WEIGHTS_PATH)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_image(image_bgr: np.ndarray):
    """
    Runs the full pipeline on a single image and returns:
      - label: 0 (authentic) or 1 (tampered)
      - forged_pct: an approximate confidence percentage that the
        image is tampered, derived from the SVM's decision function
        via a sigmoid (NOT a calibrated probability, since the
        pretrained SVM was saved with probability=False)
    """
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(_cnn_model, image_bgr)

    label = int(_svm_model.predict(feature_vector)[0])
    decision = float(_svm_model.decision_function(feature_vector)[0])

    forged_confidence = sigmoid(decision)  # classes_ = [0, 1] -> positive decision leans "1" (tampered)
    forged_pct = round(forged_confidence * 100, 1)
    real_pct = round(100 - forged_pct, 1)

    return label, real_pct, forged_pct


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image")
    except Exception:
        return jsonify({"error": "Could not read that file as an image"}), 400

    try:
        label, real_pct, forged_pct = predict_image(image_bgr)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    verdict = "Tampered" if label == 1 else "Authentic"

    return jsonify({
        "label": label,
        "verdict": verdict,
        "real_pct": real_pct,
        "forged_pct": forged_pct,
    })


load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
