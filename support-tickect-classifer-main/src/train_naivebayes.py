"""
train_naivebayes.py
-------------------
Standalone training script for Multinomial Naive Bayes on support ticket data.

Can be run directly:
    python src/train_naivebayes.py

Or imported by run_all.py:
    from src.train_naivebayes import train_naive_bayes

Pipeline:
  TF-IDF  ->  MultinomialNB
  NOTE: sublinear_tf is NOT used — MultinomialNB requires non-negative
        feature values. Raw TF or TF-IDF probabilities work correctly.

  Tuned parameters:
    - tfidf: ngram_range, max_features, min_df, use_idf
    - clf:   alpha  (Laplace smoothing — prevents zero-probability for unseen words)

Why Naive Bayes for text?
  - Generative model: learns P(word | category) from training data
  - "Naive" assumption: words are conditionally independent given the class
  - Extremely fast to train and predict
  - Strong baseline — often competitive despite its simplifying assumptions
  - alpha controls smoothing: high alpha = more smoothing, low = less
  - The reference repo used NB as its primary model (97-98% accuracy on real data)

Outputs saved:
  models/trained/naivebayes.pkl
  models/vectorizers/naivebayes_tfidf.pkl
  models/metrics/naivebayes_metrics.json
  outputs/confusion_matrices/naivebayes_cm.png
  outputs/reports/naivebayes_report.txt
  outputs/reports/naivebayes_tuning.csv
  outputs/reports/naivebayes_per_class.csv
  outputs/reports/naivebayes_top_features.txt
  outputs/reports/naivebayes_vocabulary.txt
"""

import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    # stop_words and use_idf are tuned via GridSearchCV
    # sublinear_tf is intentionally absent — MultinomialNB requires non-negative values
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   MultinomialNB()),
    ])


# ── Hyperparameter grid ────────────────────────────────────────────────────────
#
# MultinomialNB requires strictly non-negative features — no sublinear_tf.
# use_idf is explored: raw TF can outperform TF-IDF for NB on short texts.
# alpha is Laplace smoothing — prevents zero-probability for unseen n-grams.
#
# max_df trims high-frequency boilerplate terms (greetings, closings) that
# appear in almost every ticket regardless of category.
# min_df=3 is explored alongside 1 and 2: for NB the word-count model benefits
# from pruning very rare terms that are noisy rather than informative.
#
# NB fits are near-instant so the grid is intentionally wider than SVC.
#
# Word subgrid: 2*2*3*2*2*4 = 192 combos
# Char subgrid: 1*1*1*2*3   =   6 combos  →  198 total x 5 folds

def get_param_grid() -> list:
    return [
        {   # word n-gram subgrid
            "tfidf__analyzer":    ["word"],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features":[10000, 20000],
            "tfidf__min_df":      [1, 2, 3],
            "tfidf__max_df":      [0.85, 1.0],
            "tfidf__use_idf":     [True, False],
            "clf__alpha":         [1.0, 0.5, 0.1, 0.01],
        },
        {   # char_wb subgrid — char n-grams also non-negative, safe for NB
            "tfidf__analyzer":    ["char_wb"],
            "tfidf__ngram_range": [(3, 5)],
            "tfidf__max_features":[20000],
            "tfidf__use_idf":     [True, False],
            "clf__alpha":         [1.0, 0.5, 0.1],
        },
    ]


def _count_combos(grid: list) -> int:
    return sum(math.prod(len(v) for v in g.values()) for g in grid)


# ── Training entry point ──────────────────────────────────────────────────────

def train_naive_bayes(
    df=None,
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    X_all=None, y_all=None,
    class_names=None,
    label_encoder=None,
):
    """
    Train Multinomial Naive Bayes with GridSearchCV and evaluate fully.
    Returns: (best_pipeline, metrics_dict)
    """
    ensure_dirs()

    if df is None:
        print("\n" + "="*60)
        print("  MULTINOMIAL NAIVE BAYES — STANDALONE MODE")
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

    print("\n  [Multinomial NB] Building TF-IDF + MultinomialNB pipeline")
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
    print_vocabulary_stats(fitted_tfidf, model_key="naivebayes")
    save_vocabulary(fitted_tfidf, "naivebayes")
    save_vectorizer(fitted_tfidf, "naivebayes")

    print("\n  Top features per class:")
    print_top_features(best_pipeline, class_names, "naivebayes", n=10)

    # ── Full evaluation ───────────────────────────────────────────────────────

    print("\n  " + "-"*56)
    print("  EVALUATION — Multinomial Naive Bayes")
    print("  " + "-"*56)

    metrics = full_evaluation(
        best_pipeline,
        X_test, y_test,
        class_names=class_names,
        model_name="Multinomial Naive Bayes",
        model_key="naivebayes",
        X_full=X_all,
        y_full=y_all,
        cv_folds=CV_FOLDS,
    )
    metrics["best_params"] = gs.best_params_

    # ── Save artefacts ────────────────────────────────────────────────────────

    save_pipeline(best_pipeline, "naivebayes")
    save_tuning_results(gs, "naivebayes")

    return best_pipeline, metrics


if __name__ == "__main__":
    train_naive_bayes()
