"""
preprocess.py
-------------
All text cleaning and normalisation steps for support ticket bodies.

Pipeline order (mirrors reference repo's two-pass regex cleaning):
  1. Strip email header lines  (From:, Sent:, To:, CC:, Subject:)
  2. Remove URLs
  3. Remove email addresses
  4. Remove embedded image CID references  [cid:...]
  5. Remove punctuation and non-alphabetic characters
  6. Lowercase
  7. Tokenise
  8. Remove English stop words  (sklearn list — replaces repo's WordsEn.dprep)
  9. Lemmatise  (WordNetLemmatizer — reduces words to base forms)
 10. Re-join tokens into a cleaned string

The output string is what TF-IDF vectorisation receives.

NLTK data downloaded automatically on first run.
"""

import re
import os
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data quietly on first run
for _pkg in ["stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg not in ["averaged_perceptron_tagger_eng"] else f"taggers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()

# ── Regex patterns (derived from repo RegexList1 / RegexList2) ─────────────

_HEADER_RE   = re.compile(r"^(From|Sent|Received|To|CC|Subject)\s*:.*", re.IGNORECASE | re.MULTILINE)
_URL_RE      = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE    = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_CID_RE      = re.compile(r"\[cid:[^\]]*\]", re.IGNORECASE)
_NON_ALPHA   = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE = re.compile(r"\s+")


# ── Individual cleaning steps (each usable standalone) ────────────────────

def strip_email_headers(text: str) -> str:
    """Remove From:, Sent:, To:, CC:, Subject: lines."""
    return _HEADER_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_email_addresses(text: str) -> str:
    return _EMAIL_RE.sub(" ", text)


def remove_cid_tags(text: str) -> str:
    """Remove embedded image references like [cid:image001.jpg]."""
    return _CID_RE.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Keep only alphabetic characters and whitespace."""
    return _NON_ALPHA.sub(" ", text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(tokens: list) -> list:
    """Remove English stop words from a token list."""
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    """Reduce each token to its WordNet base form (lemma)."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def normalise_whitespace(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


# ── Full cleaning pipeline ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the complete preprocessing pipeline to a raw ticket body.

    Steps:
      regex cleaning  -> lowercase -> tokenise
      -> stop-word removal -> lemmatisation -> re-join
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Regex cleaning
    text = strip_email_headers(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    text = remove_cid_tags(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = normalise_whitespace(text)

    # Token-level cleaning
    tokens = text.split()
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


def show_cleaning_example(raw: str):
    """Print a before/after cleaning comparison (for learning/debug)."""
    cleaned = clean_text(raw)
    print("  RAW:")
    print(f"    {raw[:300]}")
    print("  CLEANED:")
    print(f"    {cleaned[:300]}")
    print()


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str, min_samples_per_class: int = 10) -> pd.DataFrame:
    """
    Load the support tickets CSV, apply full cleaning pipeline, and
    filter classes with too few samples.

    Returns a clean DataFrame with columns: body, category, urgency, impact.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}")

    # Show one before/after example
    print("\n  Preprocessing example:")
    show_cleaning_example(df["body"].iloc[0])

    # Apply cleaning
    df["body"] = df["body"].apply(clean_text)

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["body"].str.strip() != ""].reset_index(drop=True)
    if before - len(df):
        print(f"  Dropped {before - len(df)} empty rows after cleaning.")

    # Deduplicate on body content (mirrors repo deduplication)
    before = len(df)
    df = df.drop_duplicates(subset=["body"]).reset_index(drop=True)
    if before - len(df):
        print(f"  Dropped {before - len(df)} duplicate body rows.")

    # Filter classes with too few samples (mirrors repo min_data_per_class)
    for col in ["category", "urgency", "impact"]:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        valid  = counts[counts >= min_samples_per_class].index
        before = len(df)
        df = df[df[col].isin(valid)].reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} rows: '{col}' classes < {min_samples_per_class} samples.")

    print(f"\n  Final dataset: {len(df):,} rows")
    return df


def encode_labels(df: pd.DataFrame, target_col: str):
    """
    Encode string category labels to integers using LabelEncoder.
    Returns (encoded_array, fitted_encoder).
    The encoder is needed later to decode integer predictions back to strings.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(df[target_col].to_numpy(dtype=str))
    return encoded, le
