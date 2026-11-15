"""Flask web application for the Tweet Sentiment Analysis System.

    python app.py      ->  http://127.0.0.1:5000
"""
from flask import Flask, redirect, render_template, request, url_for

import config
from utils.predictor import get_emoji, predict_sentiment

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", max_length=config.MAX_TWEET_LENGTH, error=None)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    tweet = request.form.get("tweet", "").strip()
    if not tweet:
        return render_template(
            "index.html",
            max_length=config.MAX_TWEET_LENGTH,
            error="Please enter a tweet to analyze.",
        ), 400

    tweet = tweet[: config.MAX_TWEET_LENGTH]  # server-side guard for the 280 limit
    sentiment = predict_sentiment(tweet)
    return render_template(
        "result.html", tweet=tweet, sentiment=sentiment, emoji=get_emoji(sentiment)
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
