import os
import re
import json
import time
import joblib
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Global variables for model artifacts
MODEL = None
VECTORIZER = None
LABEL_ENCODER = None
METADATA = {}

def clean_text(text):
    """
    Exact clean_text function from Twitter_Sentiment_Analysis_ML_vs_SimpleRNN.ipynb
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#", " ", text)                            # hashtag symbol (keep word)
    text = re.sub(r"[^a-z\s]", " ", text)                     # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()                  # extra whitespace
    return text

def load_artifacts():
    global MODEL, VECTORIZER, LABEL_ENCODER, METADATA
    artifacts_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "model_artifacts"))
    
    model_path = os.path.join(artifacts_dir, "model.pkl")
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path, metadata_path]):
        print("[WARN] Warning: Artifacts not found. Run backend/train_and_save.py first.")
        return False

    MODEL = joblib.load(model_path)
    VECTORIZER = joblib.load(vectorizer_path)
    LABEL_ENCODER = joblib.load(label_encoder_path)
    
    with open(metadata_path, "r") as f:
        METADATA = json.load(f)

    print("[OK] Model artifacts successfully loaded into memory.")
    return True

# Attempt initial artifact loading
load_artifacts()

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found"}), 404
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health():
    model_status = MODEL is not None and VECTORIZER is not None
    return jsonify({
        "status": "online" if model_status else "degraded",
        "model_loaded": model_status,
        "classes": list(LABEL_ENCODER.classes_) if LABEL_ENCODER else [],
        "accuracy": METADATA.get("accuracy", 0.0) if METADATA else 0.0
    })

@app.route("/api/model-info", methods=["GET"])
def model_info():
    if not METADATA:
        return jsonify({"error": "Model metadata not available"}), 500
    return jsonify(METADATA)

@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None or VECTORIZER is None or LABEL_ENCODER is None:
        if not load_artifacts():
            return jsonify({"error": "The AI model is currently unavailable. Please run backend/train_and_save.py first."}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please enter a comment to analyze."}), 400

    raw_text = data["text"]
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Please enter a non-empty Twitter comment to analyze."}), 400

    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({
            "error": "The comment contains no recognizable words (e.g. only numbers/symbols). Please enter text."
        }), 400

    try:
        tfidf_features = VECTORIZER.transform([cleaned])
        probs = MODEL.predict_proba(tfidf_features)[0]
        classes = list(LABEL_ENCODER.classes_)

        prob_dict = {}
        for cls, prob in zip(classes, probs):
            prob_dict[cls] = round(float(prob) * 100, 1)

        top_idx = int(probs.argmax())
        top_prediction = str(classes[top_idx])
        top_confidence = round(float(probs[top_idx]) * 100, 1)

        return jsonify({
            "status": "success",
            "prediction": top_prediction,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print(f"[!] Inference error: {str(e)}")
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Sentix AI Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
