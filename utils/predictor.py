"""Load the saved model + vectoriser and predict the sentiment of a tweet."""
import sys
import time

import joblib

import config
from preprocessing import clean_text

_model = None
_vectorizer = None


def _load():
    """Load the .pkl files once; train automatically if they do not exist yet."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not (config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()):
            import train_model

            train_model.main()
        _model = joblib.load(config.MODEL_PATH)
        _vectorizer = joblib.load(config.VECTORIZER_PATH)
    return _model, _vectorizer


def predict_sentiment(text):
    """Return 'Positive', 'Negative' or 'Neutral' for a raw tweet."""
    model, vectorizer = _load()
    cleaned = clean_text(text)
    if not cleaned.strip():
        # Nothing sentiment-bearing is left after cleaning (e.g. only a URL or stop-words).
        return "Neutral"
    features = vectorizer.transform([cleaned])
    return str(model.predict(features)[0])


def get_emoji(label):
    return config.EMOJIS.get(label, "")


def measure_time(tweets):
    """Return first-call time and average time (ms) over the given tweets."""
    _load()
    times = []
    for tweet in tweets:
        start = time.perf_counter()
        predict_sentiment(tweet)
        times.append((time.perf_counter() - start) * 1000)
    return {"first_ms": times[0], "average_ms": sum(times) / len(times)}


if __name__ == "__main__":
    tweet = " ".join(sys.argv[1:]) or "I am happy today"
    label = predict_sentiment(tweet)
    print(f"{tweet!r} -> {label} {get_emoji(label)}")
