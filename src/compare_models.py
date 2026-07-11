"""
compare_models.py
-----------------
Load saved per-model metrics and produce a ranked comparison table.

Reads:
  models/metrics/<model_key>_metrics.json  (written by evaluate.py)

Writes:
  outputs/comparison_tables/model_comparison.csv
  outputs/comparison_tables/model_ranking.txt

Can be run standalone after all three train_*.py scripts have completed:
    python src/compare_models.py

Or imported by run_all.py after training all models.
"""

import os
import sys
import json
import shutil
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    MODELS_METRICS_DIR,
    MODELS_TRAINED_DIR,
    OUTPUTS_TABLES_DIR,
    ensure_dirs,
)

MODEL_KEYS = ["logreg", "linearsvc", "naivebayes"]

MODEL_DISPLAY = {
    "logreg":      "Logistic Regression",
    "linearsvc":   "LinearSVC",
    "naivebayes":  "Multinomial Naive Bayes",
}


def load_all_metrics() -> list:
    """Load JSON metrics for all trained models."""
    results = []
    for key in MODEL_KEYS:
        path = os.path.join(MODELS_METRICS_DIR, f"{key}_metrics.json")
        if not os.path.exists(path):
            print(f"  [skip] No metrics found for '{key}' — run train_{key.replace('logreg','logreg').replace('linearsvc','svm').replace('naivebayes','naivebayes')}.py first.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append(data)
    return results


def build_comparison_table(results: list) -> pd.DataFrame:
    """Build a sorted DataFrame of model metrics."""
    cols = [
        "model", "model_key", "accuracy",
        "macro_f1", "macro_precision", "macro_recall",
        "cv_mean_f1", "cv_std_f1",
    ]
    rows = []
    for r in results:
        rows.append({c: r.get(c) for c in cols})

    df = pd.DataFrame(rows)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_comparison_csv(df: pd.DataFrame) -> str:
    ensure_dirs()
    path = os.path.join(OUTPUTS_TABLES_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"  [saved] comparison table  -> outputs/comparison_tables/model_comparison.csv")
    return path


def print_ranking(df: pd.DataFrame):
    """Print the ranked comparison table to stdout."""
    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  RANK  {'Model':<28} {'Accuracy':>9} {'MacroF1':>9} {'Precision':>10} {'Recall':>8}")
    print(sep)
    for _, row in df.iterrows():
        cv_note = ""
        if row.get("cv_mean_f1") is not None:
            cv_note = f"  [CV: {row['cv_mean_f1']:.3f} +/- {row['cv_std_f1']:.3f}]"
        print(
            f"  #{int(row['rank'])}    {row['model']:<28}"
            f"  {row['accuracy']:>8.4f}"
            f"  {row['macro_f1']:>8.4f}"
            f"  {row['macro_precision']:>9.4f}"
            f"  {row['macro_recall']:>7.4f}"
            f"{cv_note}"
        )
    print(sep)


def save_ranking_txt(df: pd.DataFrame) -> str:
    ensure_dirs()
    lines = ["MODEL COMPARISON RANKING\n", "=" * 60 + "\n\n"]
    for _, row in df.iterrows():
        lines.append(f"Rank #{int(row['rank'])}  {row['model']}\n")
        lines.append(f"  Accuracy         : {row['accuracy']:.4f}  ({row['accuracy']*100:.2f}%)\n")
        lines.append(f"  Macro F1         : {row['macro_f1']:.4f}\n")
        lines.append(f"  Macro Precision  : {row['macro_precision']:.4f}\n")
        lines.append(f"  Macro Recall     : {row['macro_recall']:.4f}\n")
        if row.get("cv_mean_f1") is not None:
            lines.append(f"  CV Macro F1      : {row['cv_mean_f1']:.4f} (+/- {row['cv_std_f1']:.4f})\n")
        lines.append("\n")

    path = os.path.join(OUTPUTS_TABLES_DIR, "model_ranking.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  [saved] ranking text      -> outputs/comparison_tables/model_ranking.txt")
    return path


def select_best_model(df: pd.DataFrame) -> dict:
    """Return the top-ranked model's row as a dict and copy its pipeline."""
    best = df.iloc[0].to_dict()
    best_key  = best["model_key"]
    src  = os.path.join(MODELS_TRAINED_DIR, f"{best_key}.pkl")
    dest = os.path.join(MODELS_TRAINED_DIR, "best_model.pkl")
    if os.path.exists(src):
        shutil.copyfile(src, dest)
        print(f"  [saved] best model copy   -> models/trained/best_model.pkl  ({best['model']})")
    return best


def explain_winner(best: dict):
    """Print a short explanation of why the top model won."""
    name = best["model"]
    f1   = best["macro_f1"]
    acc  = best["accuracy"]
    print(f"\n  Why {name} ranked first (macro F1 = {f1:.4f}, accuracy = {acc*100:.2f}%):")
    if "Logistic" in name:
        print("  Logistic Regression builds a direct probability model of each")
        print("  category given the TF-IDF feature vector. L2 regularisation")
        print("  prevents overfitting on the sparse high-dimensional vocabulary,")
        print("  and the softmax output produces well-calibrated probabilities.")
    elif "SVC" in name or "LinearSVC" in name:
        print("  LinearSVC finds the maximum-margin decision boundary between")
        print("  categories in TF-IDF space. Text classification problems tend")
        print("  to be linearly separable after TF-IDF because each category")
        print("  has a distinctive vocabulary that clusters clearly.")
    elif "Bayes" in name:
        print("  Multinomial Naive Bayes models the word frequency distribution")
        print("  per category directly. Despite its independence assumption,")
        print("  it captures strong vocabulary signals and the Laplace smoothing")
        print("  (alpha) tuned by GridSearch prevents unseen-word failures.")


def run_comparison(results: list = None) -> pd.DataFrame:
    """
    Main comparison entry point. Accepts a results list (from run_all.py)
    or loads from saved JSON files if called standalone.
    """
    ensure_dirs()

    if results is None:
        print("\n" + "="*60)
        print("  MODEL COMPARISON — loading saved metrics")
        print("="*60)
        results = load_all_metrics()

    if not results:
        print("  No model metrics found. Run train_*.py scripts first.")
        return pd.DataFrame()

    df   = build_comparison_table(results)
    print_ranking(df)
    save_comparison_csv(df)
    save_ranking_txt(df)
    best = select_best_model(df)
    explain_winner(best)

    # Accuracy gate
    best_acc = best["accuracy"]
    print()
    if best_acc >= 0.80:
        print(f"  [PASS] Target accuracy >= 80% -> achieved {best_acc*100:.2f}%")
    else:
        print(f"  [WARN] Best accuracy {best_acc*100:.2f}% is below 80% target.")

    return df


if __name__ == "__main__":
    run_comparison()
