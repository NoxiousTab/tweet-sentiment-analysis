"""Central place for all file paths and constants used by the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "tweets.csv"
SAMPLE_TEST_PATH = BASE_DIR / "dataset" / "sample_test.csv"
SAMPLE_EXPECTED_PATH = BASE_DIR / "dataset" / "sample_test_expected.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

REPORTS_DIR = BASE_DIR / "reports"
ACCURACY_REPORT_PATH = REPORTS_DIR / "accuracy_report.txt"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

LABELS = ["Negative", "Neutral", "Positive"]
EMOJIS = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}

MAX_TWEET_LENGTH = 280
TEST_SIZE = 0.2
RANDOM_STATE = 42
