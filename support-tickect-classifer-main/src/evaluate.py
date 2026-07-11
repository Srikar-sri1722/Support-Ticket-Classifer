"""
evaluate.py
-----------
All evaluation functions — used by every train_*.py script.

Provides:
  - Full metric computation  (accuracy, precision, recall, F1 per class)
  - Confusion matrix PNG  (seaborn heatmap — mirrors reference repo)
  - Cross-validation scoring
  - Classification report saved to outputs/reports/
  - Per-class metric DataFrame
  - Tuning results summary from GridSearchCV
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    OUTPUTS_CM_DIR,
    OUTPUTS_REPORTS_DIR,
    MODELS_METRICS_DIR,
    ensure_dirs,
)


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred, class_names: list, model_name: str, model_key: str
) -> str:
    """
    Generate a seaborn heatmap confusion matrix and save to
    outputs/confusion_matrices/<model_key>_cm.png

    Mirrors the reference repo's seaborn-based visualisation.
    """
    ensure_dirs()
    cm  = confusion_matrix(y_true, y_pred)
    fig_size = max(6, len(class_names))
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix  -  {model_name}", fontsize=13, pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUTS_CM_DIR, f"{model_key}_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] confusion matrix -> outputs/confusion_matrices/{model_key}_cm.png")
    return path


# ── Classification report ─────────────────────────────────────────────────────

def save_classification_report(
    y_true, y_pred, class_names: list, model_key: str
) -> str:
    """Save the full sklearn classification report text to outputs/reports/."""
    ensure_dirs()
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report: {model_key}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    print(f"  [saved] classification report -> outputs/reports/{model_key}_report.txt")
    return path


# ── Per-class metric DataFrame ────────────────────────────────────────────────

def per_class_metrics(y_true, y_pred, class_names: list) -> pd.DataFrame:
    """
    Build a DataFrame with per-class precision, recall, F1 and support.
    Useful for identifying which categories are hard to classify.
    """
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in class_names:
        if cls in report_dict:
            rows.append({
                "class":     cls,
                "precision": round(report_dict[cls]["precision"], 4),
                "recall":    round(report_dict[cls]["recall"],    4),
                "f1_score":  round(report_dict[cls]["f1-score"],  4),
                "support":   int(report_dict[cls]["support"]),
            })
    return pd.DataFrame(rows)


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate(pipeline, X, y, cv_folds: int = 5, scoring: str = "f1_macro") -> dict:
    """
    Run stratified k-fold cross-validation on the full dataset.
    Returns mean, std, and individual fold scores.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_folds":    cv_folds,
        "scoring":     scoring,
        "scores":      [round(float(s), 4) for s in scores],
        "mean":        round(float(scores.mean()), 4),
        "std":         round(float(scores.std()),  4),
    }


# ── GridSearchCV tuning results ───────────────────────────────────────────────

def save_tuning_results(gs, model_key: str) -> str:
    """
    Save a summary of all GridSearchCV parameter combinations and their
    mean CV scores to outputs/reports/<model_key>_tuning.csv
    """
    ensure_dirs()
    results_df = pd.DataFrame(gs.cv_results_)
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    available = [c for c in cols if c in results_df.columns]
    results_df = results_df[available].sort_values("rank_test_score")
    results_df["mean_test_score"] = results_df["mean_test_score"].round(4)
    results_df["std_test_score"]  = results_df["std_test_score"].round(4)

    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_tuning.csv")
    results_df.to_csv(path, index=False)
    print(f"  [saved] tuning results    -> outputs/reports/{model_key}_tuning.csv")
    return path


# ── Master evaluation function ────────────────────────────────────────────────

def full_evaluation(
    pipeline,
    X_test,
    y_test,
    class_names: list,
    model_name: str,
    model_key: str,
    X_full=None,
    y_full=None,
    cv_folds: int = 5,
) -> dict:
    """
    Run the complete evaluation suite for one model:
      1. Predict on test split
      2. Compute accuracy, precision, recall, macro-F1
      3. Print full classification report
      4. Show per-class metrics DataFrame
      5. Save confusion matrix PNG
      6. Save classification report TXT
      7. Optionally run cross-validation on full data
      8. Save all scalar metrics to models/metrics/<model_key>_metrics.json

    Returns a metrics dict for use in compare_models.py
    """
    ensure_dirs()

    # 1 — Predict
    y_pred = pipeline.predict(X_test)

    # 2 — Scalar metrics
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro",     zero_division=0)
    macro_p  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_test, y_pred, average="macro",  zero_division=0)

    # 3 — Print classification report
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(f"\n  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision  : {macro_p:.4f}  (macro)")
    print(f"  Recall     : {macro_r:.4f}  (macro)")
    print(f"  F1-score   : {macro_f1:.4f}  (macro)")
    print(f"\n  Classification report:\n")
    print(report_str)

    # 4 — Per-class DataFrame
    pc_df = per_class_metrics(y_test, y_pred, class_names)
    print("  Per-class metrics:")
    print(pc_df.to_string(index=False))

    # 5+6 — Save artefacts
    plot_confusion_matrix(y_test, y_pred, class_names, model_name, model_key)
    save_classification_report(y_test, y_pred, class_names, model_key)
    pc_df.to_csv(
        os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_per_class.csv"), index=False
    )
    print(f"  [saved] per-class CSV     -> outputs/reports/{model_key}_per_class.csv")

    # 7 — Cross-validation (optional)
    cv_results = {}
    if X_full is not None and y_full is not None:
        print(f"\n  Running {cv_folds}-fold cross-validation...")
        cv_results = cross_validate(pipeline, X_full, y_full, cv_folds=cv_folds)
        print(f"  CV macro-F1: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        print(f"  Fold scores: {cv_results['scores']}")

    # 8 — Save metrics JSON
    metrics = {
        "model":           model_name,
        "model_key":       model_key,
        "accuracy":        round(acc, 6),
        "macro_f1":        round(macro_f1, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall":    round(macro_r, 6),
        "cv_mean_f1":      cv_results.get("mean", None),
        "cv_std_f1":       cv_results.get("std",  None),
        "cv_folds":        cv_folds if cv_results else None,
    }

    metrics_path = os.path.join(MODELS_METRICS_DIR, f"{model_key}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [saved] metrics JSON      -> models/metrics/{model_key}_metrics.json")

    return metrics
