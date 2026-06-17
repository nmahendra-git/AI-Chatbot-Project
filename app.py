"""
app.py - Flask Chatbot API Server
===================================
Serves the trained intent classifier via a REST API
and serves the chat UI from templates/index.html.

Endpoints:
    GET  /           → Chat UI (HTML)
    POST /chat       → { "message": "..." } → { "response": "...", "intent": "...", "confidence": 0.xx }
    GET  /health     → { "status": "ok", "model_loaded": true }
    GET  /intents    → List of all known intent tags

Usage:
    python train.py   # train first
    python app.py     # then run server
"""

import json
import os
import pickle
import random
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template, abort

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODEL_DIR       = BASE_DIR / "models"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
LABEL_ENC_PATH  = MODEL_DIR / "label_encoder.pkl"
INTENTS_PATH    = BASE_DIR / "data" / "intents.json"

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# ── Load model artifacts ──────────────────────────────────────────────────────
def load_model():
    """Load classifier and label encoder from disk."""
    if not CLASSIFIER_PATH.exists() or not LABEL_ENC_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python train.py` first."
        )
    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)
    with open(LABEL_ENC_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return classifier, label_encoder


def load_intents():
    """Load intents.json for response lookup."""
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build a tag → responses dict for O(1) lookup
    return {intent["tag"]: intent["responses"] for intent in data["intents"]}


# Globals (loaded once at startup)
try:
    classifier, label_encoder = load_model()
    intents_map = load_intents()
    MODEL_LOADED = True
    print("✅  Model loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    classifier = label_encoder = intents_map = None
    MODEL_LOADED = False


# ── Prediction helper ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.30   # below this → fallback response

def predict(text: str) -> dict:
    """
    Run inference and return:
        intent, confidence, response
    Falls back to the 'fallback' intent when confidence is low.
    """
    if not MODEL_LOADED:
        return {
            "intent": "error",
            "confidence": 0.0,
            "response": "Model not loaded. Please run `python train.py` first.",
        }

    processed = text.lower().strip()
    proba_vec  = classifier.predict_proba([processed])[0]
    best_idx   = proba_vec.argmax()
    confidence = float(proba_vec[best_idx])
    tag        = label_encoder.inverse_transform([best_idx])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        tag = "fallback"

    responses = intents_map.get(tag, intents_map.get("fallback", ["I'm not sure how to respond to that."]))
    response  = random.choice(responses)

    return {"intent": tag, "confidence": round(confidence, 4), "response": response}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: { "message": "user input text" }
    Returns: { "response": "...", "intent": "...", "confidence": 0.xx, "timestamp": "..." }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Request body must be JSON with a 'message' field."}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "'message' must not be empty."}), 400
    if len(message) > 500:
        return jsonify({"error": "'message' must not exceed 500 characters."}), 400

    result = predict(message)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return jsonify(result)


@app.route("/health")
def health():
    """Liveness / readiness check."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "intents_count": len(intents_map) if intents_map else 0,
    })


@app.route("/intents")
def list_intents():
    """Return all known intent tags (useful for debugging / docs)."""
    if not intents_map:
        return jsonify({"intents": [], "model_loaded": False})
    return jsonify({"intents": sorted(intents_map.keys()), "model_loaded": MODEL_LOADED})


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀  Starting chatbot server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
