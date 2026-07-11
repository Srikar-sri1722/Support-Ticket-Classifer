"""
train_logreg.py
---------------
Standalone training script for Logistic Regression on support ticket data.

Can be run directly:
    python src/train_logreg.py

Or imported by run_all.py:
    from src.train_logreg import train_logistic_regression

Pipeline:
  TF-IDF  ->  LogisticRegression
  Tuned parameters:
    - tfidf: ngram_range, max_features, min_df
    - clf:   C (inverse regularisation strength)

Outputs saved:
  models/trained/logreg.pkl
  models/vectorizers/logreg_tfidf.pkl
  models/metrics/logreg_metrics.json
  outputs/confusion_matrices/logreg_cm.png
  outputs/reports/logreg_report.txt
  outputs/reports/logreg_tuning.csv
  outputs/reports/logreg_per_class.csv
  outputs/reports/logreg_top_features.txt
  outputs/reports/logreg_vocabulary.txt
"""

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import load_and_clean, encode_labels
from src.evaluate   import full_evaluation, save_tuning_results
from src.features   import print_vocabulary_stats, print_top_features, save_vocabulary
from src.utils      import (
    DATA_FILE, save_pipeline, save_vectorizer, save_label_encoder,
    MODELS_TRAINED_DIR, ensure_dirs,
)
import joblib
from sklearn.model_selection import train_test_split

TARGET_COL   = "category"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MIN_SAMPLES  = 10
CV_FOLDS     = 5


# ── Pipeline ──────────────────────────────────────────────────────────────────

def get_pipeline() -> Pipeline:
    # stop_words, sublinear_tf, and solver are all tuned via GridSearchCV
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        (
            "clf",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ── Hyperparameter grid ────────────────────────────────────────────────────────
#
# Two subgrids: word n-grams and character n-grams tuned independently.
# saga solver supports multinomial loss natively; liblinear uses OvR —
# both explored to find the best fit for this 4-class problem.
#
# max_df filters terms present in >X% of documents (domain-wide boilerplate
# like "regards" or "support") that carry no discriminative signal.
# (1,3) trigrams capture multi-word expressions: "cannot access account",
# "password has expired", "server returning 500".
#
# Word subgrid: 3*2*2*2*2*5*2*2 = 960 combos
# Char subgrid: 1*1*1*1*3*1*2   =   6 combos  →  966 total x 5 folds

def get_param_grid() -> list:
    return [
        {   # word n-gram subgrid
            "tfidf__analyzer":    ["word"],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__max_features":[10000, 20000],
            "tfidf__min_df":      [1, 2],
            "tfidf__max_df":      [0.85, 1.0],
            "tfidf__sublinear_tf":[True, False],
            "clf__C":             [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__solver":        ["liblinear", "saga"],
            "clf__class_weight":  [None, "balanced"],
        },
        {   # char_wb subgrid — character n-grams within word boundaries
            # Captures: partial error codes (0x800), abbreviations (vpn, mfa, api),
            # typo-robust overlap, and morphological variants of domain terms.
            "tfidf__analyzer":    ["char_wb"],
            "tfidf__ngram_range": [(3, 5)],
            "tfidf__max_features":[20000],
            "tfidf__sublinear_tf":[True],
            "clf__C":             [0.1, 1.0, 10.0],
            "clf__solver":        ["liblinear"],
            "clf__class_weight":  [None, "balanced"],
        },
    ]


def _count_combos(grid: list) -> int:
    return sum(math.prod(len(v) for v in g.values()) for g in grid)


# ── Training entry point ──────────────────────────────────────────────────────

def train_logistic_regression(
    df=None,
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    X_all=None, y_all=None,
    class_names=None,
    label_encoder=None,
):
    """
    Train Logistic Regression with GridSearchCV and evaluate fully.

    Can be called with pre-split data (from run_all.py) or standalone
    (loads and splits data internally).

    Returns: (best_pipeline, metrics_dict)
    """
    ensure_dirs()

    # ── Load data if not provided ─────────────────────────────────────────────
    if df is None:
        print("\n" + "="*60)
        print("  LOGISTIC REGRESSION — STANDALONE MODE")
        print("="*60)
        df = load_and_clean(DATA_FILE, min_samples_per_class=MIN_SAMPLES)

    if X_train is None:
        y_encoded, label_encoder = encode_labels(df, TARGET_COL)
        class_names = [str(c) for c in label_encoder.classes_]
        X_all = df["body"].to_numpy(dtype=str)
        y_all = y_encoded

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )
        print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

        # Save label encoder when running standalone
        save_label_encoder(label_encoder)

    # ── Build and tune pipeline ───────────────────────────────────────────────

    print("\n  [Logistic Regression] Building TF-IDF + LogReg pipeline")
    grid = get_param_grid()
    print("  Hyperparameter search grid:")
    for i, g in enumerate(grid, 1):
        print(f"    Subgrid {i}:")
        for k, v in g.items():
            print(f"      {k}: {v}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_combos = _count_combos(grid)

    gs = GridSearchCV(
        get_pipeline(),
        grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=True,
    )

    print(f"\n  Fitting GridSearchCV ({n_combos} combinations x {CV_FOLDS} folds)...")
    gs.fit(X_train, y_train)

    best_pipeline = gs.best_estimator_
    print(f"\n  Best parameters  : {gs.best_params_}")
    print(f"  Best CV macro-F1 : {gs.best_score_:.4f}")

    # ── Feature inspection ────────────────────────────────────────────────────

    fitted_tfidf = best_pipeline.named_steps["tfidf"]
    print_vocabulary_stats(fitted_tfidf, model_key="logreg")
    save_vocabulary(fitted_tfidf, "logreg")
    save_vectorizer(fitted_tfidf, "logreg")

    print("\n  Top features per class:")
    print_top_features(best_pipeline, class_names, "logreg", n=10)

    # ── Full evaluation ───────────────────────────────────────────────────────

    print("\n  " + "-"*56)
    print("  EVALUATION — Logistic Regression")
    print("  " + "-"*56)

    metrics = full_evaluation(
        best_pipeline,
        X_test, y_test,
        class_names=class_names,
        model_name="Logistic Regression",
        model_key="logreg",
        X_full=X_all,
        y_full=y_all,
        cv_folds=CV_FOLDS,
    )
    metrics["best_params"] = gs.best_params_

    # ── Save artefacts ────────────────────────────────────────────────────────

    save_pipeline(best_pipeline, "logreg")
    save_tuning_results(gs, "logreg")

    return best_pipeline, metrics


if __name__ == "__main__":
    train_logistic_regression()
