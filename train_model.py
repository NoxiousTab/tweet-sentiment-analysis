"""Train the Tweet Sentiment model.

    python train_model.py

Steps: load dataset -> clean tweets -> TF-IDF -> 80/20 split -> Logistic Regression
-> evaluate -> save model/vectoriser (.pkl) and reports (txt + confusion matrix png).
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config
from preprocessing import clean_text
from utils import predictor
from utils.metrics import compute_metrics, save_reports


def main():
    df = pd.read_csv(config.DATASET_PATH)
    print(f"Loaded {len(df)} tweets:\n{df['sentiment'].value_counts().to_string()}\n")

    df["tweet"] = df["tweet"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["tweet"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["sentiment"], test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    metrics = compute_metrics(y_test, model.predict(X_test))

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    # Reload through the predictor so the timing includes the real prediction path.
    predictor._model = predictor._vectorizer = None
    sample = pd.read_csv(config.SAMPLE_TEST_PATH)["tweet"].tolist()
    timing = predictor.measure_time(sample)

    save_reports(metrics, timing, X_train.shape[0], X_test.shape[0])

    print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f} %")
    print(f"Precision: {metrics['precision'] * 100:.2f} %")
    print(f"Recall   : {metrics['recall'] * 100:.2f} %")
    print(f"F1-score : {metrics['f1'] * 100:.2f} %")
    print(f"Avg time : {timing['average_ms']:.2f} ms/tweet "
          f"(first call {timing['first_ms']:.2f} ms)\n")
    print(metrics["report"])
    print("Confusion matrix (rows = actual, cols = predicted; Negative, Neutral, Positive):")
    print(metrics["confusion"])
    print("\nSaved: model/sentiment_model.pkl, model/tfidf_vectorizer.pkl, reports/*")


if __name__ == "__main__":
    main()
