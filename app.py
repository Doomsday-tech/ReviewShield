import streamlit as st
import joblib
import json
import os
import numpy as np

from preprocess import preprocess, preprocess_steps

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

st.set_page_config(
    page_title="ReviewShield",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}

    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def predict_review(model, text):
    processed = preprocess(text)

    probs = model.predict_proba([processed])[0]

    label_idx = int(np.argmax(probs))
    confidence = float(probs[label_idx]) * 100

    label = "Suspicious Review" if label_idx == 1 else "Genuine Review"

    return label, confidence, processed


st.title("🛡️ ReviewShield")
st.caption("Fake Review Detection using NLP + Logistic Regression")

model = load_model()
metrics = load_metrics()

if model is None:
    st.error("No trained model found.")
    st.info("Run: python train.py")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:

    review = st.text_area(
        "Enter Review",
        height=220,
        placeholder="Paste a review here..."
    )

    if st.button("Analyze Review", use_container_width=True):

        if not review.strip():
            st.warning("Enter a review first.")
        else:

            label, confidence, processed = predict_review(
                model,
                review
            )

            if label == "Genuine Review":
                st.success(
                    f"✅ {label}"
                )
            else:
                st.error(
                    f"🚨 {label}"
                )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            with st.expander("View NLP Pipeline"):

                steps = preprocess_steps(review)

                st.write("### Cleaned Text")
                st.code(steps["cleaned"])

                st.write("### Tokens")
                st.write(steps["tokens"][:20])

                st.write("### Stopwords Removed")
                st.write(steps["filtered"][:20])

                st.write("### Final Processed Text")
                st.code(steps["final"])

with col2:

    st.subheader("📊 Model Metrics")

    st.metric(
        "Accuracy",
        f"{metrics.get('accuracy',0)*100:.2f}%"
    )

    st.metric(
        "Precision",
        f"{metrics.get('precision',0)*100:.2f}%"
    )

    st.metric(
        "Recall",
        f"{metrics.get('recall',0)*100:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{metrics.get('f1',0)*100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = metrics.get("confusion_matrix")

    if cm:
        st.write(cm)

st.divider()

st.caption(
    "ReviewShield • NLP • TF-IDF • Logistic Regression • Streamlit"
)