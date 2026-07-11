"""
utils.py
--------
Shared path constants, directory helpers, and model persistence.
All other src/ modules import from here so paths stay consistent.
"""

import os
import joblib

# ── Root-relative directory paths ─────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR            = os.path.join(ROOT, "data")
MODELS_TRAINED_DIR  = os.path.join(ROOT, "models", "trained")
MODELS_VECTOR_DIR   = os.path.join(ROOT, "models", "vectorizers")
MODELS_METRICS_DIR  = os.path.join(ROOT, "models", "metrics")
OUTPUTS_CM_DIR      = os.path.join(ROOT, "outputs", "confusion_matrices")
OUTPUTS_REPORTS_DIR = os.path.join(ROOT, "outputs", "reports")
OUTPUTS_TABLES_DIR  = os.path.join(ROOT, "outputs", "comparison_tables")

DATA_FILE = os.path.join(DATA_DIR, "support_tickets.csv")


def ensure_dirs():
    """Create all project output directories if they don't exist."""
    for d in [
        MODELS_TRAINED_DIR,
        MODELS_VECTOR_DIR,
        MODELS_METRICS_DIR,
        OUTPUTS_CM_DIR,
        OUTPUTS_REPORTS_DIR,
        OUTPUTS_TABLES_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


# ── Model / artefact persistence ──────────────────────────────────────────────

def save_pipeline(pipeline, model_key: str) -> str:
    """Save a fitted sklearn Pipeline to models/trained/<model_key>.pkl"""
    ensure_dirs()
    path = os.path.join(MODELS_TRAINED_DIR, f"{model_key}.pkl")
    joblib.dump(pipeline, path)
    print(f"  [saved] pipeline       -> models/trained/{model_key}.pkl")
    return path


def load_pipeline(model_key: str):
    """Load a fitted pipeline from models/trained/<model_key>.pkl"""
    path = os.path.join(MODELS_TRAINED_DIR, f"{model_key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pipeline at '{path}'. Run the corresponding train_*.py first."
        )
    return joblib.load(path)


def save_vectorizer(vectorizer, model_key: str) -> str:
    """Save a fitted TfidfVectorizer to models/vectorizers/<model_key>_tfidf.pkl"""
    ensure_dirs()
    path = os.path.join(MODELS_VECTOR_DIR, f"{model_key}_tfidf.pkl")
    joblib.dump(vectorizer, path)
    print(f"  [saved] vectorizer     -> models/vectorizers/{model_key}_tfidf.pkl")
    return path


def load_vectorizer(model_key: str):
    path = os.path.join(MODELS_VECTOR_DIR, f"{model_key}_tfidf.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No vectorizer at '{path}'.")
    return joblib.load(path)


def save_label_encoder(encoder) -> str:
    ensure_dirs()
    path = os.path.join(MODELS_TRAINED_DIR, "label_encoder.pkl")
    joblib.dump(encoder, path)
    print(f"  [saved] label encoder  -> models/trained/label_encoder.pkl")
    return path


def load_label_encoder():
    path = os.path.join(MODELS_TRAINED_DIR, "label_encoder.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No label encoder at '{path}'. Run a train_*.py first."
        )
    return joblib.load(path)


def list_trained_models() -> list:
    """Return model keys for all saved pipelines (excludes label_encoder)."""
    if not os.path.exists(MODELS_TRAINED_DIR):
        return []
    return [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(MODELS_TRAINED_DIR))
        if f.endswith(".pkl") and f != "label_encoder.pkl"
    ]
