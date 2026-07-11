"""
train_svm.py
------------
Standalone training script for LinearSVC on support ticket data.

Can be run directly:
    python src/train_svm.py

Or imported by run_all.py:
    from src.train_svm import train_svm

Pipeline:
  TF-IDF  ->  CalibratedClassifierCV(LinearSVC)
  CalibratedClassifierCV wraps LinearSVC to add predict_proba() support.
  Tuned parameters:
    - tfidf: ngram_range, max_features, min_df
    - clf:   C (SVM margin parameter)

Why LinearSVC for text?
  - Finds the maximum-margin hyperplane between classes in TF-IDF space
  - TF-IDF features are high-dimensional and sparse — ideal for SVM
  - Typically the highest-accuracy classical text classifier
  - C controls the bias-variance trade-off: low C = wider margin, more regularised

Outputs saved:
  models/trained/linearsvc.pkl
  models/vectorizers/linearsvc_tfidf.pkl
  models/metrics/linearsvc_metrics.json
  outputs/confusion_matrices/linearsvc_cm.png
  outputs/reports/linearsvc_report.txt
  outputs/reports/linearsvc_tuning.csv
  outputs/reports/linearsvc_per_class.csv
  outputs/reports/linearsvc_top_features.txt
  outputs/reports/linearsvc_vocabulary.txt
"""

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import load_and_clean, encode_labels
from src.evaluate   import full_evaluation, save_tuning_results
from src.features   import print_vocabulary_stats, print_top_features, save_vocabulary
from src.utils      import (
    DATA_FILE, save_pipeline, save_vectorizer, save_label_encoder,
    ensure_dirs,
)

TARGET_COL   = "category"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MIN_SAMPLES  = 10
CV_FOLDS     = 5


# ── Pipeline ──────────────────────────────────────────────────────────────────

def get_pipeline() -> Pipeline:
    # stop_words and sublinear_tf are tuned via GridSearchCV — not set as defaults
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        (
            "clf",
            CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
                cv=3,
            ),
        ),
    ])


# ── Hyperparameter grid ────────────────────────────────────────────────────────
#
# Two subgrids so word and char n-gram settings are tuned separately.
# char_wb (character-level n-grams within word boundaries) helps with:
#   - error codes: "0x8009030C" captures partial hex patterns
#   - abbreviations: "vpn", "mfa", "api" survive even if OOV as whole words
#   - typos: character overlap still produces useful signal
#
# max_df removes terms appearing in >X% of tickets (greeting boilerplate,
# closing phrases) that are uniform across categories and hurt discrimination.
#
# (1,3) trigrams are intentionally excluded from the SVC grid: each combo
# requires cv=3 calibration refits, so the effective multiplier per combo is 3x.
# Keeping ngram at (1,2) max preserves a practical total search time.
#
# Word subgrid: 2*2*2*2*2*4*2*2 = 512 combos
# Char subgrid: 1*1*1*1*3*1*2   =   6 combos  →  518 total x 5 folds x 3 calib

def get_param_grid() -> list:
    return [
        {   # word n-gram subgrid
            "tfidf__analyzer":              ["word"],
            "tfidf__ngram_range":           [(1, 1), (1, 2)],
            "tfidf__max_features":          [10000, 20000],
            "tfidf__min_df":                [1, 2],
            "tfidf__max_df":                [0.85, 1.0],
            "tfidf__sublinear_tf":          [True, False],
            "clf__estimator__C":            [0.01, 0.1, 1.0, 10.0],
            "clf__estimator__loss":         ["hinge", "squared_hinge"],
            "clf__estimator__class_weight": [None, "balanced"],
        },
        {   # char_wb subgrid — character n-grams within word boundaries
            "tfidf__analyzer":              ["char_wb"],
            "tfidf__ngram_range":           [(3, 5)],
            "tfidf__max_features":          [20000],
            "tfidf__sublinear_tf":          [True],
            "clf__estimator__C":            [0.1, 1.0, 10.0],
            "clf__estimator__loss":         ["squared_hinge"],
            "clf__estimator__class_weight": [None, "balanced"],
        },
    ]


def _count_combos(grid: list) -> int:
    return sum(math.prod(len(v) for v in g.values()) for g in grid)


# ── Training entry point ──────────────────────────────────────────────────────

def train_svm(
    df=None,
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    X_all=None, y_all=None,
    class_names=None,
    label_encoder=None,
):
    """
    Train LinearSVC with GridSearchCV and evaluate fully.
    Returns: (best_pipeline, metrics_dict)
    """
    ensure_dirs()

    if df is None:
        print("\n" + "="*60)
        print("  LinearSVC — STANDALONE MODE")
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
        save_label_encoder(label_encoder)

    # ── Build and tune pipeline ───────────────────────────────────────────────

    print("\n  [LinearSVC] Building TF-IDF + LinearSVC pipeline")
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
    print_vocabulary_stats(fitted_tfidf, model_key="linearsvc")
    save_vocabulary(fitted_tfidf, "linearsvc")
    save_vectorizer(fitted_tfidf, "linearsvc")

    print("\n  Top features per class:")
    print_top_features(best_pipeline, class_names, "linearsvc", n=10)

    # ── Full evaluation ───────────────────────────────────────────────────────

    print("\n  " + "-"*56)
    print("  EVALUATION — LinearSVC")
    print("  " + "-"*56)

    metrics = full_evaluation(
        best_pipeline,
        X_test, y_test,
        class_names=class_names,
        model_name="LinearSVC",
        model_key="linearsvc",
        X_full=X_all,
        y_full=y_all,
        cv_folds=CV_FOLDS,
    )
    metrics["best_params"] = gs.best_params_

    # ── Save artefacts ────────────────────────────────────────────────────────

    save_pipeline(best_pipeline, "linearsvc")
    save_tuning_results(gs, "linearsvc")

    return best_pipeline, metrics


if __name__ == "__main__":
    train_svm()
