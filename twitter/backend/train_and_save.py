import os
import re
import json
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
np.random.seed(SEED)

def clean_text(text):
    """
    Exact clean_text function from Twitter_Sentiment_Analysis_ML_vs_SimpleRNN.ipynb
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#", " ", text)                            # hashtag symbol (keep word)
    text = re.sub(r"[^a-z\s]", " ", text)                     # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()                  # extra whitespace
    return text

def find_dataset():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "twitter_training.csv"),
        os.path.join(os.path.dirname(__file__), "twitter_training.csv"),
        r"C:\Users\Zuneha\Downloads\twitter_training.csv\twitter_training.csv",
        r"C:\Users\Zuneha\Downloads\twitter_training.csv",
        r"C:\Users\Zuneha\Desktop\twitter\twitter_training.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"[+] Found dataset at: {path}")
            return path
    raise FileNotFoundError("Could not find twitter_training.csv. Please ensure it exists.")

def main():
    print("=" * 60)
    print("Sentix AI — Model Training & Artifact Generation Pipeline")
    print("=" * 60)

    dataset_path = find_dataset()
    print("[1/5] Loading twitter dataset...")
    df = pd.read_csv(dataset_path, header=None, names=["id", "entity", "sentiment", "text"])
    print(f"    Raw dataset shape: {df.shape}")

    print("[2/5] Cleaning & preprocessing text...")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"    Shape after cleaning: {df.shape}")

    print("[3/5] Encoding labels...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    classes = list(le.classes_)
    class_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print(f"    Classes: {class_mapping}")

    print("[4/5] Train/Test Split & TF-IDF Vectorization...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    print("    Fitting TF-IDF Vectorizer (max_features=20000, ngram_range=(1, 2))...")
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    print(f"    TF-IDF Train Matrix Shape: {X_train_tfidf.shape}")

    print("[5/5] Training Classifier (Logistic Regression)...")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs", n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    train_time = round(time.time() - start_time, 2)

    preds = model.predict(X_test_tfidf)
    acc = float(accuracy_score(y_test, preds))
    print(f"\n[OK] Model Test Accuracy: {acc * 100:.2f}% (Trained in {train_time}s)")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=classes))

    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts"))
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "model.pkl")
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")

    print("\nSaving artifacts...")
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(le, label_encoder_path)

    metadata = {
        "model_type": "Logistic Regression (TF-IDF)",
        "accuracy": round(acc * 100, 2),
        "dataset_name": "Twitter Entity Sentiment Analysis (twitter_training.csv)",
        "total_samples": int(df.shape[0]),
        "vocab_size": 20000,
        "ngram_range": [1, 2],
        "classes": classes,
        "class_mapping": class_mapping,
        "preprocessing_steps": [
            "Lowercase conversion",
            "URL removal (http/https/www)",
            "User mention removal (@username)",
            "Hashtag symbol strip (retaining hashtag text)",
            "Punctuation & digit removal",
            "Whitespace collapse & trimming"
        ],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Artifacts successfully saved to: {artifacts_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
