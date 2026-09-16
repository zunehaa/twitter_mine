# Sentix AI — Twitter Sentiment Intelligence Web Application 🚀

**Sentix AI** is a premium, full-stack AI web application that analyzes Twitter comments in real-time and classifies their sentiment into **Positive**, **Negative**, **Neutral**, or **Irrelevant** using an actual trained machine learning classifier built on TF-IDF textual feature vectors.

---

## 🌟 Key Features

- **10/10 Modern Dark AI Laboratory Aesthetic**: Custom dark design system featuring obsidian glassmorphism, responsive grid layouts, ambient canvas glows, and subtle micro-interactions.
- **Genuine Machine Learning Model**: Uses the exact preprocessing, TF-IDF vectorization (`max_features=20000`, `ngram_range=(1,2)`), and fitted classifier trained on ~74,000 Twitter comments (`twitter_training.csv`).
- **Real Probability Distributions**: Calculates genuine, un-fabricated confidence scores and probability breakdowns for all 4 sentiment classes.
- **Sample Tweets Bar**: Quick-test pills for instant sentiment prediction across all classes.
- **Session Analytics & Live History**: Local storage session tracking displaying total comments analyzed, sentiment class distributions, and a clearable history table.
- **RESTful Flask API**: Asynchronous backend serving inference and model metadata endpoints.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, Vanilla CSS3 (Custom Dark Theme, Flexbox/Grid), Modern JavaScript (ES6+ fetch API).
- **Backend API**: Python 3.10+, Flask REST Server.
- **Machine Learning**: `scikit-learn` (`LogisticRegression`, `TfidfVectorizer`, `LabelEncoder`), `pandas`, `numpy`, `joblib`.
- **Dataset**: Twitter Entity Sentiment Analysis (`twitter_training.csv`).

---

## 📁 Project Structure

```text
twitter/
├── frontend/
│   ├── index.html        # Main web interface
│   ├── style.css         # Custom dark laboratory design system
│   └── script.js         # Interactive application logic & API connection
│
├── backend/
│   ├── app.py            # Flask REST API server (Inference & metadata endpoints)
│   └── train_and_save.py # Model training script using exact notebook preprocessing
│
├── model_artifacts/      # Generated trained artifacts
│   ├── model.pkl         # Fitted Logistic Regression Classifier
│   ├── vectorizer.pkl   # Fitted TF-IDF Vectorizer
│   ├── label_encoder.pkl# Fitted LabelEncoder
│   └── metadata.json     # Model specs, vocabulary size, and accuracy metrics
│
├── Twitter_Sentiment_Analysis_ML_vs_SimpleRNN.ipynb # Reference training notebook
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🧠 Preprocessing & Inference Pipeline

The inference pipeline reproduces the exact data cleaning and feature engineering specified in `Twitter_Sentiment_Analysis_ML_vs_SimpleRNN.ipynb`:

1. **Text Cleaning**:
   - Lowercasing text.
   - Removing URLs (`http://...`, `www...`).
   - Stripping user mentions (`@user`).
   - Cleaning hashtag symbols while keeping tag text (`#word` &rarr; `word`).
   - Removing punctuation and digits.
   - Collapsing extra whitespace.

2. **Vectorization**:
   - Transforming cleaned text using fitted `TfidfVectorizer(max_features=20000, ngram_range=(1, 2))`.

3. **Classification & Decoding**:
   - Computing class probability distribution via `model.predict_proba()`.
   - Decoding class integers using `LabelEncoder` (`Irrelevant`, `Negative`, `Neutral`, `Positive`).

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Open your terminal inside the project directory and run:
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Step 1: Train & Save Model Artifacts (Optional if already trained)
To train the classifier on `twitter_training.csv` and generate `model_artifacts/`:
```bash
python backend/train_and_save.py
```
*Outputs `model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`, and `metadata.json` into `model_artifacts/`.*

### Step 2: Start the Flask API Server
Run the backend web server:
```bash
python backend/app.py
```
The server will start at: **`http://127.0.0.1:5000`**

### Step 3: Open in Browser
Navigate to **`http://127.0.0.1:5000`** in your web browser to use **Sentix AI**!

---

## 📡 API Reference

### `POST /api/predict`
Analyzes a Twitter comment and returns sentiment prediction.

**Request Body**:
```json
{
  "text": "This update is breathtaking, super smooth performance!"
}
```

**Response**:
```json
{
  "status": "success",
  "prediction": "Positive",
  "confidence": 89.4,
  "probabilities": {
    "Positive": 89.4,
    "Neutral": 6.2,
    "Irrelevant": 2.5,
    "Negative": 1.9
  },
  "raw_text": "This update is breathtaking, super smooth performance!",
  "cleaned_text": "this update is breathtaking super smooth performance",
  "timestamp": "2026-09-16 20:37:00"
}
```

### `GET /api/model-info`
Returns model training metadata, dataset size, and accuracy metrics.

### `GET /api/health`
Returns backend health status and model loading state.

---

## 🛡️ License & Acknowledgments
Built for Twitter Sentiment Intelligence demonstration based on the Twitter Entity Sentiment Analysis dataset.
