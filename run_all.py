"""
run_all.py
----------
Master training script — runs the full ML experimentation pipeline.

Executes in order:
  1. Load and preprocess dataset            (src/preprocess.py)
  2. Shared train/test split
  3. Train Logistic Regression              (src/train_logreg.py)
  4. Train LinearSVC                        (src/train_svm.py)
  5. Train Multinomial Naive Bayes          (src/train_naivebayes.py)
  6. Compare all models and select best     (src/compare_models.py)

Each model can also be trained and evaluated independently:
    python src/train_logreg.py
    python src/train_svm.py
    python src/train_naivebayes.py

Run this script:
    python run_all.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess       import load_and_clean, encode_labels
from src.train_logreg     import train_logistic_regression
from src.train_svm        import train_svm
from src.train_naivebayes import train_naive_bayes
from src.compare_models   import run_comparison
from src.utils            import DATA_FILE, save_label_encoder, ensure_dirs

TARGET_COL   = "category"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MIN_SAMPLES  = 10


def main():
    ensure_dirs()

    print("\n" + "="*60)
    print("  NLP SUPPORT TICKET CLASSIFIER - FULL TRAINING PIPELINE")
    print("="*60)

    # ── Step 1: Load and preprocess ───────────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 1 — DATA LOADING & PREPROCESSING")
    print("="*60)

    df = load_and_clean(DATA_FILE, min_samples_per_class=MIN_SAMPLES)

    print(f"\n  Dataset shape    : {df.shape}")
    print(f"  Target column    : {TARGET_COL}")
    print(f"  Class distribution:")
    print(df[TARGET_COL].value_counts().to_string())

    # ── Step 2: Encode labels and split ──────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 2 — LABEL ENCODING & TRAIN/TEST SPLIT")
    print("="*60)

    y_encoded, label_encoder = encode_labels(df, TARGET_COL)
    class_names = [str(c) for c in label_encoder.classes_]
    X_all = df["body"].to_numpy(dtype=str)
    y_all = y_encoded

    save_label_encoder(label_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    print(f"\n  Classes          : {class_names}")
    print(f"  Encoded mapping  : {dict(enumerate(class_names))}")
    print(f"\n  Total samples    : {len(X_all)}")
    print(f"  Training split   : {len(X_train)}  ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  Test split       : {len(X_test)}   ({TEST_SIZE*100:.0f}%)")
    print(f"  Stratified       : Yes — class proportions preserved")

    # Shared kwargs passed to each training function
    shared = dict(
        df=df,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X_all=X_all, y_all=y_all,
        class_names=class_names,
        label_encoder=label_encoder,
    )

    # ── Step 3: Logistic Regression ───────────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 3 — LOGISTIC REGRESSION")
    print("="*60)
    _, logreg_metrics = train_logistic_regression(**shared)

    # ── Step 4: LinearSVC ─────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 4 — LinearSVC")
    print("="*60)
    _, svm_metrics = train_svm(**shared)

    # ── Step 5: Multinomial Naive Bayes ───────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 5 — MULTINOMIAL NAIVE BAYES")
    print("="*60)
    _, nb_metrics = train_naive_bayes(**shared)

    # ── Step 6: Model comparison ──────────────────────────────────────────────

    print("\n" + "="*60)
    print("  STEP 6 — MODEL COMPARISON & BEST MODEL SELECTION")
    print("="*60)

    all_results = [logreg_metrics, svm_metrics, nb_metrics]
    comparison_df = run_comparison(results=all_results)

    # ── Final summary ─────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print("\n  Saved models:")
    print("    models/trained/logreg.pkl")
    print("    models/trained/linearsvc.pkl")
    print("    models/trained/naivebayes.pkl")
    print("    models/trained/best_model.pkl")
    print("    models/trained/label_encoder.pkl")
    print("\n  Saved vectorizers:")
    print("    models/vectorizers/logreg_tfidf.pkl")
    print("    models/vectorizers/linearsvc_tfidf.pkl")
    print("    models/vectorizers/naivebayes_tfidf.pkl")
    print("\n  Evaluation outputs:")
    print("    outputs/confusion_matrices/*.png")
    print("    outputs/reports/*_report.txt")
    print("    outputs/reports/*_tuning.csv")
    print("    outputs/reports/*_per_class.csv")
    print("    outputs/reports/*_top_features.txt")
    print("    outputs/comparison_tables/model_comparison.csv")
    print("    outputs/comparison_tables/model_ranking.txt")
    print("\n  Launch UI:")
    print("    streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
