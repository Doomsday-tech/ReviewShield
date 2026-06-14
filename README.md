# 🛡️ ReviewShield

**NLP-based Fake Review Detection System**

ReviewShield classifies product reviews as **Genuine** or **Fake** using a machine learning pipeline built with scikit-learn and NLTK, served through a clean dark-themed Streamlit interface.

---

## Features

| Feature | Details |
|---|---|
| **Detection** | Binary classification — Genuine vs Fake |
| **Confidence Score** | Probability output shown as a percentage |
| **NLP Pipeline** | Lowercase → Tokenize → Stopword Removal → Lemmatize → TF-IDF |
| **Models** | Logistic Regression and Naive Bayes (best is auto-selected) |
| **Analytics** | Accuracy, Precision, Recall, F1, Confusion Matrix |
| **UI** | Dark-themed Streamlit app with pipeline breakdown panel |

---

## Project Structure

```
ReviewShield/
│
├── data/               ← Place your dataset CSV here (optional)
├── models/
│   └── metrics.json    ← Auto-generated after training
│
├── app.py              ← Streamlit application
├── train.py            ← Model training script
├── preprocess.py       ← NLP preprocessing pipeline
├── model.pkl           ← Saved model (auto-generated after training)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone / Download the project

```bash
cd ReviewShield
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add your own dataset

Place a CSV file named `reviews.csv` inside the `data/` folder.

Expected columns:

| Column | Description |
|---|---|
| `review` | Raw review text |
| `label` | `0` = Genuine, `1` = Fake |

**Recommended public datasets:**

- [Fake Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- [Amazon Fake Reviews — Kaggle](https://www.kaggle.com/datasets/lievgarcia/amazon-reviews)
- [Yelp Review Dataset — Yelp](https://www.yelp.com/dataset)

If no CSV is found, ReviewShield automatically uses a high-quality built-in synthetic dataset.

### 5. Train the model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train Logistic Regression and Naive Bayes
- Select the best model by F1 score
- Save `model.pkl` and `models/metrics.json`

### 6. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Tip:** You can also train directly from the UI by clicking **Train Model Now** on first launch.

---

## NLP Pipeline

Each review passes through the following steps before classification:

```
Raw Text
   │
   ▼ Lowercase + Remove punctuation / numbers
   │
   ▼ Tokenize (NLTK word_tokenize)
   │
   ▼ Remove stopwords (NLTK English stopwords)
   │
   ▼ Lemmatize (WordNetLemmatizer)
   │
   ▼ TF-IDF Vectorization (unigrams + bigrams, top 10 000 features)
   │
   ▼ Classifier (Logistic Regression or Naive Bayes)
```

Click **View NLP Pipeline Steps** in the app to see each transformation applied to your input.

---

## Model Details

| Model | Vectorizer | Notes |
|---|---|---|
| Logistic Regression | TF-IDF (1-2 grams) | `C=1.0`, `max_iter=1000` |
| Naive Bayes | TF-IDF (1-2 grams) | `alpha=0.5` (Laplace smoothing) |

The model with the higher **F1 score** on the test split is saved automatically.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML pipeline, TF-IDF, classifiers, metrics
- **NLTK** — Tokenization, stopwords, lemmatization
- **Pandas / NumPy** — Data handling
- **Streamlit** — Web UI
- **Joblib** — Model serialisation
- **Matplotlib / Seaborn** — Confusion matrix plot

---

## Screenshots

```
┌────────────────────────────────────────────────────────────┐
│  🛡️  ReviewShield                                          │
│  NLP-powered fake review detection                         │
├──────────────────────────────┬─────────────────────────────┤
│  🔍 Analyze a Review         │  📊 Model Performance       │
│  ┌────────────────────────┐  │  Accuracy   Precision       │
│  │ Enter review text …    │  │  94.2%      93.8%           │
│  └────────────────────────┘  │  Recall     F1 Score        │
│  [🛡️ Analyze Review]         │  94.6%      94.2%           │
│                              │                             │
│  ✅ Genuine Review           │  [Confusion Matrix]         │
│  Confidence: 91.4%  ████░    │                             │
│                              │  🤖 Model Comparison        │
│  [🔬 NLP Pipeline Steps]     │  ★ Logistic Regression      │
│                              │    Naive Bayes              │
└──────────────────────────────┴─────────────────────────────┘
```

---

## Example Reviews

| Review | Expected |
|---|---|
| "Solid product, works as described. Arrived on time, no issues after two months." | ✅ Genuine |
| "ABSOLUTELY AMAZING!!! Changed my life COMPLETELY!!! Buy this NOW!!!" | 🚨 Fake |
| "Good value for the price. A few minor scratches on arrival but overall fine." | ✅ Genuine |
| "BEST PRODUCT EVER!!! I told EVERYONE to buy this immediately!!!" | 🚨 Fake |

---

## License

MIT — free to use, modify, and distribute.
