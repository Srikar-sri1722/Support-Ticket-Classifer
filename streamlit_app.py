"""
streamlit_app.py
----------------
Full ML experimentation dashboard for the support ticket classifier.

Tabs:
  1. Predict         — classify any ticket using any trained model
  2. All Models      — full evaluation results for every model side by side
  3. Model Comparison — ranked table, CV scores, best model explanation

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.predict import load_predictor, predict_ticket, compare_models_on_text
from src.utils import (
    MODELS_TRAINED_DIR, MODELS_METRICS_DIR,
    OUTPUTS_CM_DIR, OUTPUTS_REPORTS_DIR, OUTPUTS_TABLES_DIR,
    list_trained_models,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
)

MODEL_DISPLAY = {
    "best_model":  "Best Model (auto-selected)",
    "logreg":      "Logistic Regression",
    "linearsvc":   "LinearSVC",
    "naivebayes":  "Multinomial Naive Bayes",
}

TRAINED_MODELS = list_trained_models()


# ── Check models exist ────────────────────────────────────────────────────────

if not TRAINED_MODELS:
    st.error("No trained models found. Run `python run_all.py` first.")
    st.stop()


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_key: str):
    return load_predictor(model_key)


def load_report(model_key: str) -> str:
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_report.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "Report not found. Run training first."


def load_per_class(model_key: str) -> pd.DataFrame | None:
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_per_class.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_tuning(model_key: str) -> pd.DataFrame | None:
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_tuning.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_metrics(model_key: str) -> dict:
    path = os.path.join(MODELS_METRICS_DIR, f"{model_key}_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_top_features(model_key: str) -> str:
    path = os.path.join(OUTPUTS_REPORTS_DIR, f"{model_key}_top_features.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return ""


def cm_path(model_key: str) -> str | None:
    p = os.path.join(OUTPUTS_CM_DIR, f"{model_key}_cm.png")
    return p if os.path.exists(p) else None


def prob_chart(all_probs: dict):
    classes = list(all_probs.keys())
    probs   = [all_probs[c] for c in classes]
    colors  = ["#1d4ed8" if p == max(probs) else "#93c5fd" for p in probs]
    fig, ax = plt.subplots(figsize=(6, max(2, len(classes) * 0.6)))
    ax.barh(classes, probs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability")
    ax.set_title("Category Probabilities")
    for i, (p, c) in enumerate(zip(probs, classes)):
        ax.text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Support Ticket Classifier")
st.caption(
    "Full ML experimentation pipeline — Logistic Regression, LinearSVC, "
    "Multinomial Naive Bayes — with TF-IDF vectorisation and GridSearchCV tuning. "
    "Inspired by the Endava support-tickets-classification repository."
)

tab_predict, tab_models, tab_compare = st.tabs([
    "Predict", "All Models", "Model Comparison"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Classify a Support Ticket")

        ticket_text = st.text_area(
            "Paste ticket body here",
            placeholder=(
                "Example:\nHi IT Support,\n\nI am unable to connect to the VPN "
                "since this morning. Error: Authentication failed — code 0x8009030C. "
                "I have restarted my machine and the issue persists.\n\nKind regards"
            ),
            height=200,
            key="ticket_input",
        )

        model_opts = {
            MODEL_DISPLAY.get(m, m): m
            for m in ["best_model"] + [m for m in TRAINED_MODELS if m != "best_model"]
        }
        chosen_display = st.selectbox(
            "Select model", list(model_opts.keys()), key="predict_model"
        )
        chosen_key = model_opts[chosen_display]

        predict_btn       = st.button("Predict Category", type="primary", use_container_width=True)
        compare_all_btn   = st.button("Compare All Models on This Text", use_container_width=True)

    with col_right:
        st.subheader("Preprocessing Pipeline")
        st.markdown("""
**Steps applied to ticket text:**
1. Strip email headers (`From:`, `Sent:`, `To:`, `CC:`)
2. Remove URLs
3. Remove email addresses
4. Remove punctuation and numbers
5. Lowercase
6. Remove English stop words
7. Lemmatise (WordNetLemmatizer)
        """)

    # ── Single model prediction ───────────────────────────────────────────────

    if predict_btn:
        if not ticket_text.strip():
            st.warning("Enter some ticket text first.")
        else:
            predictor = get_predictor(chosen_key)
            result    = predict_ticket(predictor, ticket_text)

            if result["category"] == "unknown":
                st.error("Text became empty after cleaning. Add more descriptive content.")
            else:
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Category", result["category"])
                m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                m3.metric("Model", MODEL_DISPLAY.get(result["model_key"], result["model_key"]))

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = prob_chart(result["all_probabilities"])
                    st.pyplot(fig)
                    plt.close(fig)
                with col_b:
                    st.markdown("**Preprocessed text (what the model sees)**")
                    st.code(result["cleaned_text"], language=None)

    # ── Compare all models ────────────────────────────────────────────────────

    if compare_all_btn:
        if not ticket_text.strip():
            st.warning("Enter some ticket text first.")
        else:
            st.divider()
            st.subheader("All Models on This Text")
            all_results = compare_models_on_text(ticket_text)
            if not all_results:
                st.warning("No models found. Run `python run_all.py` first.")
            else:
                cols = st.columns(len(all_results))
                for col, res in zip(cols, all_results):
                    with col:
                        st.markdown(f"**{MODEL_DISPLAY.get(res['model_key'], res['model_key'])}**")
                        st.metric("Category",   res["category"])
                        st.metric("Confidence", f"{res['confidence']*100:.1f}%")
                        for cls, prob in sorted(
                            res["all_probabilities"].items(), key=lambda x: -x[1]
                        ):
                            st.progress(prob, text=f"{cls}: {prob*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALL MODELS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_models:
    st.subheader("Full Evaluation — All Three Models")
    st.caption(
        "Confusion matrices, classification reports, per-class metrics, "
        "and top TF-IDF features for each model."
    )

    display_keys = [m for m in ["logreg", "linearsvc", "naivebayes"] if m in TRAINED_MODELS]

    if not display_keys:
        st.info("No individual model results found. Run `python run_all.py` first.")
    else:
        for model_key in display_keys:
            model_name = MODEL_DISPLAY.get(model_key, model_key)
            metrics    = load_metrics(model_key)

            with st.expander(f"{model_name}", expanded=(model_key == display_keys[0])):

                # Metrics row
                if metrics:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",   f"{metrics.get('accuracy', 0)*100:.2f}%")
                    c2.metric("Macro F1",   f"{metrics.get('macro_f1', 0):.4f}")
                    c3.metric("Precision",  f"{metrics.get('macro_precision', 0):.4f}")
                    c4.metric("Recall",     f"{metrics.get('macro_recall', 0):.4f}")
                    if metrics.get("cv_mean_f1"):
                        st.caption(
                            f"5-fold CV macro-F1: "
                            f"{metrics['cv_mean_f1']:.4f} "
                            f"(+/- {metrics.get('cv_std_f1', 0):.4f})"
                        )

                col_cm, col_report = st.columns([1, 1])

                # Confusion matrix
                with col_cm:
                    cm_p = cm_path(model_key)
                    if cm_p:
                        st.markdown("**Confusion Matrix**")
                        st.image(cm_p, use_container_width=True)
                    else:
                        st.info("Confusion matrix not found.")

                # Classification report
                with col_report:
                    st.markdown("**Classification Report**")
                    report = load_report(model_key)
                    st.code(report, language=None)

                # Per-class metrics table
                pc_df = load_per_class(model_key)
                if pc_df is not None:
                    st.markdown("**Per-class Metrics**")
                    st.dataframe(pc_df, hide_index=True, use_container_width=True)

                # Top features
                top_feats = load_top_features(model_key)
                if top_feats:
                    with st.expander("Top TF-IDF features per category"):
                        st.code(top_feats, language=None)

                # Tuning results
                tuning_df = load_tuning(model_key)
                if tuning_df is not None:
                    with st.expander("GridSearchCV tuning results (all parameter combinations)"):
                        st.dataframe(tuning_df.head(20), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.subheader("Model Comparison Table")

    table_path = os.path.join(OUTPUTS_TABLES_DIR, "model_comparison.csv")
    ranking_path = os.path.join(OUTPUTS_TABLES_DIR, "model_ranking.txt")

    if os.path.exists(table_path):
        df = pd.read_csv(table_path)

        # Display table
        display_df = df[["rank", "model", "accuracy", "macro_f1",
                          "macro_precision", "macro_recall",
                          "cv_mean_f1", "cv_std_f1"]].copy()
        display_df["accuracy"] = (display_df["accuracy"] * 100).round(2).astype(str) + "%"
        display_df.columns = [
            "Rank", "Model", "Accuracy", "Macro F1",
            "Precision", "Recall", "CV F1 (mean)", "CV F1 (std)"
        ]
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Bar chart comparison
        st.divider()
        st.subheader("Visual Comparison")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        raw_df = pd.read_csv(table_path)
        metrics_to_plot = [
            ("accuracy",      "Accuracy"),
            ("macro_f1",      "Macro F1"),
            ("macro_precision","Macro Precision"),
        ]
        for ax, (col, label) in zip(axes, metrics_to_plot):
            colors = ["#1d4ed8" if i == 0 else "#93c5fd" for i in range(len(raw_df))]
            ax.bar(raw_df["model"].str.replace(" ", "\n"), raw_df[col], color=colors)
            ax.set_title(label)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            for i, v in enumerate(raw_df[col]):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Best model info
        st.divider()
        best = raw_df.iloc[0]
        st.subheader(f"Best Model: {best['model']}")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Accuracy", f"{best['accuracy']*100:.2f}%")
        bc2.metric("Macro F1", f"{best['macro_f1']:.4f}")
        bc3.metric("CV F1",    f"{best.get('cv_mean_f1', 'N/A')}")

        if os.path.exists(ranking_path):
            with st.expander("Full ranking text"):
                st.code(open(ranking_path, encoding="utf-8").read(), language=None)

    else:
        st.info("No comparison table found. Run `python run_all.py` to generate it.")

    # Architecture summary
    st.divider()
    st.subheader("Pipeline Architecture")
    st.code("""
Dataset (support_tickets.csv)
        |
        v
  preprocess.py
    - strip email headers (From:, Sent:, To:, CC:)
    - remove URLs and email addresses
    - remove punctuation / non-alphabetic chars
    - lowercase
    - remove English stop words (NLTK)
    - lemmatise (WordNetLemmatizer)
        |
        v
  80 / 20 stratified train-test split
        |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
  train_logreg.py           train_svm.py            train_naivebayes.py
  TF-IDF +                  TF-IDF +                TF-IDF +
  LogisticRegression        CalibratedLinearSVC     MultinomialNB
  GridSearchCV              GridSearchCV            GridSearchCV
        |                           |                           |
        v                           v                           v
  evaluate.py  (accuracy, precision, recall, F1, confusion matrix, CV)
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    v
                          compare_models.py
                          model_comparison.csv
                          best_model.pkl
                                    |
                                    v
                          streamlit_app.py  (this UI)
    """, language=None)
