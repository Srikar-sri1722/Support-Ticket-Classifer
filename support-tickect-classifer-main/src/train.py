"""
train.py
--------
Complete ML experimentation pipeline for support ticket classification.

Workflow:
  1.  Load & inspect dataset
  2.  Clean and preprocess ticket body text
  3.  Encode target labels
  4.  Train/test split (stratified 80/20)
  5.  Logistic Regression  — TF-IDF + GridSearchCV + full evaluation
  6.  LinearSVC            — TF-IDF + GridSearchCV + full evaluation
  7.  Multinomial NB       — TF-IDF + GridSearchCV + full evaluation
  8.  Model comparison table
  9.  Best model selection + saved pipelines

All three models are saved individually.
All confusion matrices and classification reports are saved to outputs/.

Run:
    python src/train.py
"""

import os
import sys
import shutil
import warnings
import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import load_and_clean, encode_labels
from src.utils import plot_confusion_matrix, save_pipeline, save_comparison_table, print_comparison_table

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ROOT         = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH    = os.path.join(ROOT, "data", "support_tickets.csv")
MODELS_DIR   = os.path.join(ROOT, "models")
OUTPUTS_DIR  = os.path.join(ROOT, "outputs")

TARGET_COL   = "category"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MIN_SAMPLES  = 10
CV_FOLDS     = 5

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save_report(report_str: str, model_key: str):
    """Save a classification_report string to outputs/<model_key>_report.txt."""
    path = os.path.join(OUTPUTS_DIR, f"{model_key}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"  Saved classification report -> {path}")
    return path


def save_metrics(metrics: dict, model_key: str):
    """Save per-model scalar metrics to outputs/<model_key>_metrics.csv."""
    path = os.path.join(OUTPUTS_DIR, f"{model_key}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"  Saved metrics              -> {path}")
    return path


def run_gridsearch(pipeline, param_grid, X_train, y_train, cv):
    """Fit GridSearchCV, print best params and CV score, return best estimator."""
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params (CV)   : {gs.best_params_}")
    print(f"  Best CV macro-F1   : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def evaluate(pipeline, X_test, y_test, class_names, model_name, model_key):
    """
    Run full evaluation on the test split.
    Prints accuracy, macro-F1, and full classification report.
    Saves confusion matrix PNG and classification report TXT.
    Returns a metrics dict.
    """
    y_pred = pipeline.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro",     zero_division=0)
    macro_p  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_test, y_pred, average="macro",  zero_division=0)

    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"\n  Test accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Test macro-F1        : {macro_f1:.4f}")
    print(f"  Test macro-Precision : {macro_p:.4f}")
    print(f"  Test macro-Recall    : {macro_r:.4f}")
    print(f"\n  Classification report:\n")
    print(report)

    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    save_report(report, model_key)

    metrics = {
        "model":     model_name,
        "model_key": model_key,
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "macro_precision": macro_p,
        "macro_recall":    macro_r,
    }
    save_metrics(metrics, model_key)
    return metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def run_training():

    section("STEP 1 — LOAD & INSPECT DATASET")

    df = load_and_clean(DATA_PATH, min_samples_per_class=MIN_SAMPLES)

    print(f"  Dataset shape    : {df.shape}")
    print(f"  Columns          : {list(df.columns)}")
    print(f"\n  Target column    : {TARGET_COL}")
    print(f"  Class counts:\n{df[TARGET_COL].value_counts().to_string()}")
    print(f"\n  Sample body (cleaned):\n  {df['body'].iloc[0][:200]}")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — ENCODE LABELS
    # ─────────────────────────────────────────────────────────────────────────

    section("STEP 2 — LABEL ENCODING")

    y, label_encoder = encode_labels(df, TARGET_COL)
    class_names = [str(c) for c in label_encoder.classes_]

    print(f"  Original labels  : {list(df[TARGET_COL].unique())}")
    print(f"  Encoded mapping  :")
    for idx, name in enumerate(class_names):
        print(f"    {idx}  ->  {name}")

    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\n  Saved label encoder -> models/label_encoder.pkl")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — TRAIN / TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────────

    section("STEP 3 — TRAIN / TEST SPLIT  (80 / 20, stratified)")

    X = df["body"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"  Total samples    : {len(X)}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Stratify=True    : class ratios preserved in both splits")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"  Cross-validation : StratifiedKFold, k={CV_FOLDS}")

    all_results = []


    # =========================================================================
    # STEP 4 — LOGISTIC REGRESSION
    # =========================================================================

    section("STEP 4 — LOGISTIC REGRESSION")

    print("""
  Why Logistic Regression?
  - Discriminative linear model trained directly on class boundaries
  - L2 regularisation (C parameter) prevents overfitting on sparse TF-IDF features
  - Outputs calibrated probabilities via softmax (multi-class)
  - Interpretable: feature weights show which words drive each category
    """)

    logreg_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",   # remove common English words (the, is, at...)
                sublinear_tf=True,      # apply 1 + log(tf) — compresses high-frequency terms
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                multi_class="auto",
            ),
        ),
    ])

    logreg_param_grid = {
        "tfidf__ngram_range":  [(1, 1), (1, 2)],   # unigrams vs unigrams+bigrams
        "tfidf__max_features": [20000, 50000],       # vocabulary size cap
        "tfidf__min_df":       [1, 2],               # min document frequency per term
        "clf__C":              [0.1, 1.0, 10.0],     # inverse regularisation strength
    }

    print(f"  TF-IDF parameters searched : {list(logreg_param_grid.keys())[:3]}")
    print(f"  Classifier parameters      : C in {logreg_param_grid['clf__C']}")
    print(f"  Grid size                  : {2*2*2*3} = 24 combinations x {CV_FOLDS} folds")
    print()

    logreg_best, logreg_params = run_gridsearch(
        logreg_pipeline, logreg_param_grid, X_train, y_train, cv
    )

    logreg_metrics, logreg_preds = evaluate(
        logreg_best, X_test, y_test, class_names,
        model_name="Logistic Regression",
        model_key="logreg",
    )

    save_pipeline(logreg_best, "logreg")
    logreg_metrics["best_params"] = logreg_params
    all_results.append(logreg_metrics)


    # =========================================================================
    # STEP 5 — LinearSVC
    # =========================================================================

    section("STEP 5 — LinearSVC  (Support Vector Machine)")

    print("""
  Why LinearSVC?
  - Finds the maximum-margin hyperplane separating classes in TF-IDF feature space
  - Particularly effective on high-dimensional sparse text data
  - Typically the fastest and most accurate classical classifier on NLP tasks
  - C parameter controls the margin width / misclassification trade-off
  - Wrapped in CalibratedClassifierCV to produce probability scores
    """)

    linearsvc_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
                cv=3,
            ),
        ),
    ])

    linearsvc_param_grid = {
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "tfidf__max_features": [20000, 50000],
        "tfidf__min_df":       [1, 2],
        "clf__estimator__C":   [0.01, 0.1, 1.0],   # SVM margin parameter
    }

    print(f"  TF-IDF parameters searched : {list(linearsvc_param_grid.keys())[:3]}")
    print(f"  Classifier parameters      : C in {linearsvc_param_grid['clf__estimator__C']}")
    print(f"  Grid size                  : {2*2*2*3} = 24 combinations x {CV_FOLDS} folds")
    print()

    linearsvc_best, linearsvc_params = run_gridsearch(
        linearsvc_pipeline, linearsvc_param_grid, X_train, y_train, cv
    )

    linearsvc_metrics, linearsvc_preds = evaluate(
        linearsvc_best, X_test, y_test, class_names,
        model_name="LinearSVC",
        model_key="linearsvc",
    )

    save_pipeline(linearsvc_best, "linearsvc")
    linearsvc_metrics["best_params"] = linearsvc_params
    all_results.append(linearsvc_metrics)


    # =========================================================================
    # STEP 6 — MULTINOMIAL NAIVE BAYES
    # =========================================================================

    section("STEP 6 — MULTINOMIAL NAIVE BAYES")

    print("""
  Why Multinomial Naive Bayes?
  - Generative probabilistic model: learns P(word | class) from training data
  - Assumes word counts are conditionally independent given the class (naive)
  - Naturally suited to TF-IDF / count features — requires non-negative values
  - alpha (Laplace smoothing) prevents zero-probability for unseen words
  - Fast to train; strong baseline for text classification (mirrors reference repo)
  - use_idf=True/False is searched: repo found IDF weighting sometimes hurts NB
    """)

    naivebayes_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                # sublinear_tf NOT used — MultinomialNB requires non-negative features
                # and raw term frequencies / TF-IDF probabilities work better here
            ),
        ),
        (
            "clf",
            MultinomialNB(),
        ),
    ])

    naivebayes_param_grid = {
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "tfidf__max_features": [20000, 50000],
        "tfidf__min_df":       [1, 2],
        "tfidf__use_idf":      [True, False],              # mirrors reference repo grid
        "clf__alpha":          [1.0, 0.1, 0.01, 0.001],   # mirrors reference repo grid
    }

    print(f"  TF-IDF parameters searched : {list(naivebayes_param_grid.keys())[:4]}")
    print(f"  Classifier parameters      : alpha in {naivebayes_param_grid['clf__alpha']}")
    print(f"  Grid size                  : {2*2*2*2*4} = 64 combinations x {CV_FOLDS} folds")
    print()

    naivebayes_best, naivebayes_params = run_gridsearch(
        naivebayes_pipeline, naivebayes_param_grid, X_train, y_train, cv
    )

    naivebayes_metrics, naivebayes_preds = evaluate(
        naivebayes_best, X_test, y_test, class_names,
        model_name="Multinomial Naive Bayes",
        model_key="naivebayes",
    )

    save_pipeline(naivebayes_best, "naivebayes")
    naivebayes_metrics["best_params"] = naivebayes_params
    all_results.append(naivebayes_metrics)


    # =========================================================================
    # STEP 7 — MODEL COMPARISON
    # =========================================================================

    section("STEP 7 — MODEL COMPARISON TABLE")

    comparison_df = save_comparison_table(all_results)
    print_comparison_table(comparison_df)

    # Save full comparison with all metrics
    full_cols = ["model", "accuracy", "macro_f1", "macro_precision", "macro_recall", "best_params"]
    available_cols = [c for c in full_cols if c in comparison_df.columns]
    comparison_df[available_cols].to_csv(
        os.path.join(OUTPUTS_DIR, "model_comparison_full.csv"), index=False
    )
    print(f"  Saved full comparison -> outputs/model_comparison_full.csv")


    # =========================================================================
    # STEP 8 — BEST MODEL SELECTION
    # =========================================================================

    section("STEP 8 — BEST MODEL SELECTION")

    best_row = comparison_df.iloc[0]   # sorted descending by macro_f1

    print(f"  Winner           : {best_row['model']}")
    print(f"  Accuracy         : {best_row['accuracy']*100:.2f}%")
    print(f"  Macro F1         : {best_row['macro_f1']:.4f}")

    # Explain why best model won
    print(f"\n  Why {best_row['model']} performed best:")
    if "Logistic" in best_row["model"]:
        print("  Logistic Regression found the strongest linear decision boundary")
        print("  in TF-IDF feature space. L2 regularisation prevented overfitting")
        print("  on the high-dimensional but sparse vocabulary.")
    elif "SVC" in best_row["model"] or "LinearSVC" in best_row["model"]:
        print("  LinearSVC found the widest margin separating ticket categories.")
        print("  SVM excels on high-dimensional sparse text data — each category")
        print("  has distinctive vocabulary that creates clean linear boundaries.")
    elif "Bayes" in best_row["model"]:
        print("  Naive Bayes captured strong per-category word probability signals.")
        print("  With Laplace smoothing tuned by GridSearch, it correctly modelled")
        print("  the conditional word distributions for each ticket type.")

    # Copy best model
    best_key  = best_row["model_key"]
    best_src  = os.path.join(MODELS_DIR, f"{best_key}.pkl")
    best_dest = os.path.join(MODELS_DIR, "best_model.pkl")
    shutil.copyfile(best_src, best_dest)

    # Save best model metadata
    pd.DataFrame([{
        "model":     best_row["model"],
        "model_key": best_key,
        "accuracy":  best_row["accuracy"],
        "macro_f1":  best_row["macro_f1"],
    }]).to_csv(os.path.join(OUTPUTS_DIR, "best_model_info.csv"), index=False)

    print(f"\n  Saved as         : models/best_model.pkl")
    print(f"  Metadata saved   : outputs/best_model_info.csv")

    # Final accuracy gate
    print()
    if best_row["accuracy"] >= 0.80:
        print(f"  [PASS] Target accuracy >= 80%  ->  achieved {best_row['accuracy']*100:.2f}%")
    else:
        print(f"  [WARN] Accuracy {best_row['accuracy']*100:.2f}% is below the 80% target.")

    print("\n" + "="*60)
    print("  ALL STEPS COMPLETE — models saved to models/")
    print("  All outputs saved to outputs/")
    print("="*60)

    return comparison_df


if __name__ == "__main__":
    run_training()
