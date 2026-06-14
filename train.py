
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocess import preprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "Reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Reviews.csv not found.\nExpected location:\n{DATA_PATH}"
        )

    print("Loading Amazon Reviews dataset...")

    df = pd.read_csv(DATA_PATH)

    required_cols = ["Text", "Score"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset."
            )

    df = df[["Text", "Score"]].dropna()

    # Convert to binary labels
    # 0 = Genuine
    # 1 = Fake / Suspicious

    df["label"] = df["Score"].apply(
        lambda score: 1 if score <= 2 else 0
    )

    df = df.rename(
        columns={
            "Text": "review"
        }
    )

    return df[["review", "label"]]


def train():
    print("\n=== REVIEWSHIELD TRAINING STARTED ===\n")

    df = load_data()

    print(f"Rows loaded: {len(df):,}")

    print("Preprocessing reviews...")

    df["processed"] = (
        df["review"]
        .astype(str)
        .apply(preprocess)
    )

    X = df["processed"]
    y = df["label"]

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training Logistic Regression model...")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== TRAINING COMPLETE ===\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nFiles created:")
    print(f"Model   : {MODEL_PATH}")
    print(f"Metrics : {METRICS_PATH}")


if __name__ == "__main__":
    train()

