"""
features.py
-----------
TF-IDF feature engineering utilities.

Responsibilities:
  - Build configurable TfidfVectorizer instances
  - Fit vectorizers on training data and save them
  - Analyse vocabulary: size, top terms, document frequencies
  - Compare n-gram configurations
  - Extract per-class top features from a fitted pipeline
  - Print feature statistics for learning/inspection
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import save_vectorizer, OUTPUTS_REPORTS_DIR, ensure_dirs


# ── Default TF-IDF configurations per model type ──────────────────────────────

TFIDF_CONFIGS = {
    "logreg": {
        "stop_words":   "english",
        "sublinear_tf": True,       # log(1+tf) — dampens high-frequency terms
        "ngram_range":  (1, 1),
        "max_features": 30000,
        "min_df":       1,
    },
    "linearsvc": {
        "stop_words":   "english",
        "sublinear_tf": True,
        "ngram_range":  (1, 1),
        "max_features": 30000,
        "min_df":       1,
    },
    "naivebayes": {
        "stop_words":   "english",
        "sublinear_tf": False,      # NB needs non-negative raw frequencies
        "ngram_range":  (1, 1),
        "max_features": 30000,
        "min_df":       1,
        "use_idf":      True,
    },
}


def build_vectorizer(config: dict) -> TfidfVectorizer:
    """Instantiate a TfidfVectorizer from a config dict."""
    return TfidfVectorizer(**config)


def fit_and_save_vectorizer(X_train, model_key: str, config: dict = None) -> TfidfVectorizer:
    """
    Fit a TfidfVectorizer on X_train and save it separately.
    Used for inspection — the Pipeline inside each train_*.py also fits
    its own vectorizer internally to prevent leakage during GridSearchCV.
    """
    cfg = config or TFIDF_CONFIGS.get(model_key, TFIDF_CONFIGS["logreg"])
    vectorizer = build_vectorizer(cfg)
    vectorizer.fit(X_train)
    save_vectorizer(vectorizer, model_key)
    return vectorizer


# ── Vocabulary analysis ────────────────────────────────────────────────────────

def vocabulary_stats(vectorizer: TfidfVectorizer) -> dict:
    """Return summary statistics for a fitted vectorizer's vocabulary."""
    vocab_size = len(vectorizer.vocabulary_)
    feature_names = vectorizer.get_feature_names_out()

    stats = {
        "vocabulary_size":  vocab_size,
        "ngram_range":      vectorizer.ngram_range,
        "max_features":     vectorizer.max_features,
        "min_df":           vectorizer.min_df,
        "use_idf":          getattr(vectorizer, "use_idf", True),
        "sublinear_tf":     getattr(vectorizer, "sublinear_tf", False),
        "sample_features":  list(feature_names[:20]),
    }
    return stats


def print_vocabulary_stats(vectorizer: TfidfVectorizer, model_key: str = ""):
    """Print vocabulary statistics to stdout."""
    stats = vocabulary_stats(vectorizer)
    prefix = f"[{model_key}] " if model_key else ""
    print(f"\n  {prefix}TF-IDF Vocabulary Statistics")
    print(f"  {'-'*40}")
    print(f"  Vocabulary size  : {stats['vocabulary_size']:,}")
    print(f"  N-gram range     : {stats['ngram_range']}")
    print(f"  Max features     : {stats['max_features']}")
    print(f"  Min doc freq     : {stats['min_df']}")
    print(f"  Use IDF          : {stats['use_idf']}")
    print(f"  Sublinear TF     : {stats['sublinear_tf']}")
    print(f"  Sample features  : {stats['sample_features']}")


def save_vocabulary(vectorizer: TfidfVectorizer, model_key: str):
    """Save the full vocabulary list to outputs/reports/<model_key>_vocabulary.txt"""
    ensure_dirs()
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_vocabulary.txt")
    features = vectorizer.get_feature_names_out()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Vocabulary for: {model_key}\n")
        f.write(f"Total terms: {len(features)}\n\n")
        for term in features:
            f.write(term + "\n")
    print(f"  [saved] vocabulary     -> outputs/reports/{model_key}_vocabulary.txt")


# ── Per-class top features ────────────────────────────────────────────────────

def get_top_features_per_class(pipeline, class_names: list, n: int = 15) -> dict:
    """
    Extract the top n TF-IDF features for each class from a fitted pipeline.

    Works with Logistic Regression (coef_) and LinearSVC (estimator.coef_).
    Returns a dict: {class_name: [(feature, weight), ...]}

    Note: MultinomialNB uses feature_log_prob_ instead — handled separately.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]
    features   = vectorizer.get_feature_names_out()
    result     = {}

    # Logistic Regression
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        for i, class_name in enumerate(class_names):
            top_idx = np.argsort(coef[i])[::-1][:n]
            result[class_name] = [(features[j], round(float(coef[i][j]), 4)) for j in top_idx]

    # CalibratedClassifierCV wrapping LinearSVC
    elif hasattr(clf, "calibrated_classifiers_"):
        base = clf.calibrated_classifiers_[0].estimator
        if hasattr(base, "coef_"):
            coef = base.coef_
            for i, class_name in enumerate(class_names):
                top_idx = np.argsort(coef[i])[::-1][:n]
                result[class_name] = [(features[j], round(float(coef[i][j]), 4)) for j in top_idx]

    # MultinomialNB — use log probabilities
    elif hasattr(clf, "feature_log_prob_"):
        log_prob = clf.feature_log_prob_
        for i, class_name in enumerate(class_names):
            top_idx = np.argsort(log_prob[i])[::-1][:n]
            result[class_name] = [(features[j], round(float(log_prob[i][j]), 4)) for j in top_idx]

    return result


def print_top_features(pipeline, class_names: list, model_key: str, n: int = 10):
    """Print and save per-class top features."""
    top = get_top_features_per_class(pipeline, class_names, n=n)
    if not top:
        print("  Feature extraction not supported for this classifier type.")
        return

    lines = [f"Top {n} features per class — {model_key}\n{'='*50}\n"]
    for cls, feats in top.items():
        print(f"\n  [{cls}] top {n} terms:")
        lines.append(f"\n[{cls}]\n")
        for term, weight in feats:
            print(f"    {term:<30}  {weight:+.4f}")
            lines.append(f"  {term:<30}  {weight:+.4f}\n")

    # Save to file
    ensure_dirs()
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_top_features.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n  [saved] top features   -> outputs/reports/{model_key}_top_features.txt")


# ── N-gram comparison ─────────────────────────────────────────────────────────

def compare_ngram_configs(X_train, X_test, y_train, y_test, configs: list = None) -> pd.DataFrame:
    """
    Fit a simple Logistic Regression on different TF-IDF n-gram configs and
    compare accuracy. Returns a DataFrame of results.

    configs — list of dicts with TfidfVectorizer kwargs to compare.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    if configs is None:
        configs = [
            {"ngram_range": (1, 1), "max_features": 20000, "stop_words": "english"},
            {"ngram_range": (1, 2), "max_features": 20000, "stop_words": "english"},
            {"ngram_range": (2, 2), "max_features": 20000, "stop_words": "english"},
            {"ngram_range": (1, 3), "max_features": 20000, "stop_words": "english"},
        ]

    rows = []
    for cfg in configs:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**cfg)),
            ("clf",   LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append({
            "ngram_range":     str(cfg.get("ngram_range", (1, 1))),
            "max_features":    cfg.get("max_features", "all"),
            "accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "macro_f1":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "vocab_size":      len(pipe.named_steps["tfidf"].vocabulary_),
        })

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df
