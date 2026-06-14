"""
preprocess.py — ReviewShield NLP Preprocessing Pipeline
Handles text cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """Download required NLTK data files silently."""
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for name in packages:
        try:
            nltk.download(name, quiet=True)
        except Exception:
            pass


# Run on import so the pipeline is always ready
download_nltk_resources()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase and strip non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Split text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove common English stopwords."""
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def lemmatize(tokens: list) -> list:
    """Reduce each token to its dictionary base form."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """
    Full NLP pipeline:
      1. Clean  ->  2. Tokenize  ->  3. Remove stopwords  ->  4. Lemmatize
    Returns a single reconstructed string ready for TF-IDF vectorization.
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_steps(text: str) -> dict:
    """
    Return each intermediate step so the UI can display the pipeline.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize(filtered)
    final = " ".join(lemmatized)

    return {
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "lemmatized": lemmatized,
        "final": final,
    }
