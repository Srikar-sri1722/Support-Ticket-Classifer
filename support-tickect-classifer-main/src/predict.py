"""
predict.py
----------
Inference module — completely separate from training code.

Mirrors the reference repository's webservice.py inference flow but as a
simple local function rather than a REST endpoint.

Workflow:
  1. Load any saved pipeline from models/trained/
  2. Apply the same clean_text() preprocessing used during training
  3. Call pipeline.predict() — TF-IDF vectorisation is inside the pipeline
  4. Call pipeline.predict_proba() for confidence scores
  5. Decode integer prediction back to category string

Usage (CLI):
    python src/predict.py "I cannot connect to the VPN."
    python src/predict.py "My account is locked." linearsvc

Usage (import):
    from src.predict import load_predictor, predict_ticket, predict_batch
    predictor = load_predictor("logreg")
    result = predict_ticket(predictor, raw_text)
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import clean_text
from src.utils import MODELS_TRAINED_DIR, list_trained_models, load_label_encoder


def load_predictor(model_key: str = "best_model") -> dict:
    """
    Load a saved pipeline and its LabelEncoder.

    Returns a predictor dict:
      pipeline      — fitted sklearn Pipeline (TF-IDF + classifier)
      label_encoder — LabelEncoder for decoding integer predictions
      model_key     — identifier of the loaded model
      class_names   — list of category label strings
    """
    pipeline_path = os.path.join(MODELS_TRAINED_DIR, f"{model_key}.pkl")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"No model at '{pipeline_path}'. Run the corresponding train_*.py first."
        )

    pipeline      = joblib.load(pipeline_path)
    label_encoder = load_label_encoder()
    class_names   = [str(c) for c in label_encoder.classes_]

    return {
        "pipeline":      pipeline,
        "label_encoder": label_encoder,
        "model_key":     model_key,
        "class_names":   class_names,
    }


def predict_ticket(predictor: dict, raw_text: str) -> dict:
    """
    Predict the category of a single support ticket.

    Preprocessing applied here is identical to training:
      clean_text()  ->  pipeline.predict()  ->  decode label

    Returns:
      category          — predicted category string
      confidence        — probability of the top prediction (0-1)
      all_probabilities — {class_name: probability} for all classes
      cleaned_text      — what the model actually received
      model_key         — which model was used
    """
    pipeline      = predictor["pipeline"]
    label_encoder = predictor["label_encoder"]
    class_names   = predictor["class_names"]

    cleaned = clean_text(raw_text)

    if not cleaned.strip():
        return {
            "category":          "unknown",
            "confidence":        0.0,
            "all_probabilities": {},
            "cleaned_text":      cleaned,
            "model_key":         predictor["model_key"],
        }

    pred_encoded = pipeline.predict([cleaned])[0]
    category     = label_encoder.inverse_transform([pred_encoded])[0]

    proba      = pipeline.predict_proba([cleaned])[0]
    all_probs  = {name: round(float(p), 4) for name, p in zip(class_names, proba)}
    confidence = float(proba[pred_encoded])

    return {
        "category":          str(category),
        "confidence":        confidence,
        "all_probabilities": all_probs,
        "cleaned_text":      cleaned,
        "model_key":         predictor["model_key"],
    }


def predict_batch(predictor: dict, texts: list) -> list:
    """
    Predict categories for a list of ticket texts.
    Returns a list of result dicts (same structure as predict_ticket).
    """
    return [predict_ticket(predictor, t) for t in texts]


def compare_models_on_text(raw_text: str) -> list:
    """
    Run all available saved models on the same input text.
    Returns a list of result dicts sorted by confidence descending.
    """
    available = list_trained_models()
    results   = []
    for key in available:
        if key == "best_model":
            continue
        try:
            predictor = load_predictor(key)
            result    = predict_ticket(predictor, raw_text)
            result["model_display"] = key
            results.append(result)
        except FileNotFoundError:
            continue
    return sorted(results, key=lambda r: r["confidence"], reverse=True)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<ticket text>\" [model_key]")
        print("       model_key options:", list_trained_models())
        sys.exit(1)

    raw_text  = sys.argv[1]
    model_key = sys.argv[2] if len(sys.argv) > 2 else "best_model"

    predictor = load_predictor(model_key)
    result    = predict_ticket(predictor, raw_text)

    print(f"\n  Model            : {result['model_key']}")
    print(f"  Cleaned input    : {result['cleaned_text'][:120]}...")
    print(f"\n  Predicted cat.   : {result['category']}")
    print(f"  Confidence       : {result['confidence']*100:.1f}%")
    print("\n  All probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"    {cls:<25}  {prob*100:5.1f}%  {bar}")
