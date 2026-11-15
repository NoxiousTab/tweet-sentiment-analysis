from app import app
from utils.predictor import get_emoji, predict_sentiment


def test_prediction_is_one_of_three_classes():
    assert predict_sentiment("I am happy") in {"Positive", "Negative", "Neutral"}


def test_demo_tweets():
    assert predict_sentiment("I am Happy.") == "Positive"
    assert predict_sentiment("I don't Have time") == "Negative"
    assert predict_sentiment("meeting starts on 10 am") == "Neutral"


def test_text_with_no_words_left_is_neutral():
    assert predict_sentiment("https://t.co/abc") == "Neutral"


def test_emoji_mapping():
    assert get_emoji("Positive") == "😊"
    assert get_emoji("Negative") == "😞"
    assert get_emoji("Neutral") == "😐"


def test_web_pages():
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    resp = client.post("/predict", data={"tweet": "I am Happy."})
    assert resp.status_code == 200
    assert b"Positive Sentiment" in resp.data


def test_empty_tweet_is_rejected():
    resp = app.test_client().post("/predict", data={"tweet": "   "})
    assert resp.status_code == 400
    assert b"Please enter a tweet" in resp.data
