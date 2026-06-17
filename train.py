"""
train.py - Train the Intent Classification Model
=================================================
Trains a scikit-learn TF-IDF + Logistic Regression classifier
on the intents.json dataset and saves the model artifacts.

Usage:
    python train.py

Outputs (saved to models/):
    - classifier.pkl   : Trained classifier pipeline
    - label_encoder.pkl: Label encoder for intent tags
    - training_report.txt: Accuracy + classification report
"""

import json
import pickle
import os
import random
from pathlib import Path

# ── Dependencies ────────────────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install scikit-learn numpy\n"
        "then try again."
    )

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "intents.json"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASSIFIER_PATH    = MODEL_DIR / "classifier.pkl"
LABEL_ENC_PATH     = MODEL_DIR / "label_encoder.pkl"
REPORT_PATH        = MODEL_DIR / "training_report.txt"


# ── 1. Load & parse intents ──────────────────────────────────────────────────
def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            texts.append(pattern.lower().strip())
            labels.append(tag)

    print(f"[Data]  Loaded {len(texts)} samples across "
          f"{len(set(labels))} intents")
    return texts, labels, data["intents"]


# ── 2. Build & train pipeline ────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=5.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train(texts, labels):
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # Guard: skip split when dataset is tiny
    if len(set(labels)) > 1 and len(texts) > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = texts, texts, y, y
        print("[Train] Small dataset – training on full set")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    )

    return pipeline, le, acc, report


# ── 3. Save artifacts ────────────────────────────────────────────────────────
def save_artifacts(pipeline, le, acc, report):
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENC_PATH, "wb") as f:
        pickle.dump(le, f)

    summary = (
        f"Accuracy : {acc:.4f} ({acc*100:.2f}%)\n"
        f"Model    : TF-IDF (1-2 grams) + Logistic Regression\n"
        f"Intents  : {len(le.classes_)}\n\n"
        + "=" * 50 + "\nClassification Report\n" + "=" * 50 + "\n"
        + report
    )
    with open(REPORT_PATH, "w") as f:
        f.write(summary)

    print(f"[Save]  Classifier   → {CLASSIFIER_PATH}")
    print(f"[Save]  LabelEncoder → {LABEL_ENC_PATH}")
    print(f"[Save]  Report       → {REPORT_PATH}")
    return summary


# ── 4. Quick sanity check ────────────────────────────────────────────────────
def smoke_test(pipeline, le, intents_data):
    samples = [
        "Hello there!",
        "What's your name?",
        "Tell me a joke",
        "Goodbye",
        "I need some help",
        "thanks a lot",
    ]
    print("\n[Test]  Smoke test predictions:")
    for text in samples:
        enc_label = pipeline.predict([text.lower()])[0]
        tag = le.inverse_transform([enc_label])[0]

        # grab a random response for that tag
        response = "???"
        for intent in intents_data:
            if intent["tag"] == tag and intent["responses"]:
                response = random.choice(intent["responses"])
                break

        proba = pipeline.predict_proba([text.lower()]).max()
        print(f"  '{text}' → [{tag}] (conf={proba:.2f})  →  \"{response}\"")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print(" Chatbot Intent Classifier – Training Script")
    print("=" * 50)

    texts, labels, intents_data = load_data(DATA_PATH)
    pipeline, le, acc, report   = train(texts, labels)
    summary = save_artifacts(pipeline, le, acc, report)

    print(f"\n[Result] Accuracy: {acc*100:.2f}%")
    smoke_test(pipeline, le, intents_data)

    print("\n✅  Training complete! Run `python app.py` to start the server.")
