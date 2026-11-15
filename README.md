# Tweet Sentiment Analysis System

Mini project for **410245A – Information Retrieval**
Modern Education Society's Wadia College of Engineering, Pune · Department of Computer Engineering · AY 2026-27, Sem-VII
Faculty: Prof. V.D. Gaikwad · Class: BE COMP 3

A small NLP + machine-learning system that reads a tweet and classifies it as **Positive 😊**, **Negative 😞** or **Neutral 😐**, served through a Flask web app.

```
raw tweet → clean (lowercase, strip URL/@/#/digits/punctuation) → NLTK tokenise + stop-words
          → TF-IDF vector → Logistic Regression → sentiment + emoji
```

| Part | Technology |
|---|---|
| Language | Python 3.x |
| Web | Flask, HTML5, CSS3, JavaScript |
| NLP | NLTK (tokeniser, stop-words), `re`, `string` |
| Features | scikit-learn `TfidfVectorizer` |
| Model | scikit-learn `LogisticRegression`, saved with `joblib` (`.pkl`) |
| Data / plots | Pandas, NumPy, Matplotlib, Seaborn |
| Tests | pytest |

---

## 1. Setup (do this once, before the demo day)

```bash
git clone https://github.com/NoxiousTab/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis

python -m venv venv
# Windows:      venv\Scripts\activate
# Linux / Mac:  source venv/bin/activate

pip install -r requirements.txt
python setup_nltk.py        # optional: downloads NLTK punkt + stopwords (needs internet)
```

`setup_nltk.py` is optional. If the NLTK data is missing, the project automatically uses NLTK's regex tokenizer and a built-in copy of the NLTK English stop-word list, so the results are the same.

## 2. Train the model

```bash
python train_model.py
```

This loads `dataset/tweets.csv` (30 tweets: 10 Positive, 10 Negative, 10 Neutral), cleans them, builds TF-IDF features, splits 80/20 (`random_state=42`), trains Logistic Regression (`max_iter=1000`) and writes:

- `model/sentiment_model.pkl` and `model/tfidf_vectorizer.pkl`
- `reports/accuracy_report.txt`, `reports/classification_report.txt`, `reports/confusion_matrix.png`

Trained model files are already included in the repo, so training is only needed if you change the dataset or code. (If you see a scikit-learn version warning on another machine, just run `python train_model.py` again.)

## 3. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in the browser.

| Route | What it does |
|---|---|
| `/` | Home page: tweet box (max 280 characters), live character counter, **Analyze Sentiment** button |
| `/predict` | Cleans the tweet, predicts, shows the result page with emoji |
| `/about` | About Project page |

## 4. Run the tests

```bash
pytest -v
```

11 tests covering text cleaning, prediction and the web routes.

---

## 5. Demo script for the teacher (≈ 5–7 minutes)

### Step 1 – Introduce the problem (30 s)
> "Millions of tweets are posted daily. Reading them by hand is slow and doesn't scale, so we built a system that automatically tells whether a tweet is positive, negative or neutral."

### Step 2 – Show the project structure (30 s)
Open the folder in VS Code and point out: `preprocessing/` (cleaning), `train_model.py` (training), `utils/predictor.py` (prediction), `utils/metrics.py` (evaluation), `app.py` (Flask), `templates/` + `static/` (UI), `tests/`.

### Step 3 – Show the preprocessing (1 min)
```bash
python -c "from preprocessing import clean_text; print(clean_text('Loving the new phone!!! @riya #awesome https://t.co/xyz 100%'))"
```
Expected output: `loving new phone awesome`
> "Lower-casing, URL / @mention / # / digit / punctuation removal, NLTK tokenisation and stop-word removal."

### Step 4 – Train live (1 min)
```bash
python train_model.py
```
Show the printed metrics, then open `reports/confusion_matrix.png`.
> "TF-IDF gives each word a weight – high if frequent in a tweet but rare across the dataset. Logistic Regression learns which weights point to which sentiment."

### Step 5 – Live web demo (2 min)
```bash
python app.py
```
Open http://127.0.0.1:5000 and type these tweets (these are the same ones used in the report screenshots):

| Type this | Expected result |
|---|---|
| `I am Happy.` | 😊 Positive Sentiment |
| `I don't Have time` | 😞 Negative Sentiment |
| `meeting starts on 10 am` | 😐 Neutral Sentiment |

Also show: the live **0 / 280 characters** counter, and clicking *Analyze Sentiment* with an empty box (the browser asks you to fill the field).

### Step 6 – Batch prediction on `sample_test.csv` (1 min)
```bash
python predict_sample.py
```
Real output from this repo:

```
 No  Tweet                                        Expected  Predicted   Result
 1   I absolutely love this new phone             Positive  Positive    Correct
 2   This product is very bad and disappointing   Negative  Negative    Correct
 3   The meeting is scheduled at 4 PM             Neutral   Neutral     Correct
 4   I am very happy with the service             Positive  Positive    Correct
 5   This movie was boring and terrible           Negative  Negative    Correct
 6   The weather is normal today                  Neutral   Neutral     Correct
 7   I really enjoyed this application            Positive  Positive    Correct
 8   The customer support was useless             Negative  Negative    Correct
 9   My exam is tomorrow morning                  Neutral   Neutral     Correct
10   I got a new laptop yesterday                 Neutral   Positive    Incorrect
11   I hate this website design                   Negative  Neutral     Incorrect
12   This is the best experience ever             Positive  Negative    Incorrect
13   The train will arrive at 8 PM                Neutral   Neutral     Correct
14   I am not satisfied with this purchase        Negative  Negative    Correct
15   Today is just another normal day             Neutral   Positive    Incorrect

11 of 15 classified in agreement with the manual label (73.3%)
```
(The "expected" labels were assigned manually – the file itself has no labels.)

### Step 7 – Be upfront about limitations (1 min)
> "With only 30 training tweets, the test set is 6 tweets, so one tweet changes accuracy by ~17 points. The numbers describe the tiny dataset, not the method. With Sentiment140 (1.6 M tweets) accuracy would improve a lot."

### Step 8 – Future work (30 s)
Large dataset (Sentiment140), n-gram features and negation handling, Naive Bayes / SVM / Random Forest comparison, LSTM / BERT, live tweets via the X API, emoji + multi-language support, batch analysis with dashboards.

---

## 6. Results from this repo (what you will see when you train)

| Metric | Value |
|---|---|
| Training / test tweets | 24 / 6 |
| Accuracy | 50.00 % |
| Precision (weighted) | 61.11 % |
| Recall (weighted) | 50.00 % |
| F1-score (weighted) | 52.22 % |
| Avg. processing time | well under 1 ms per tweet |

> **Important – match your report to your demo.** The PDF report lists a 16.67 % test accuracy and 10/15 on the sample file, which came from the dataset you originally used. The tweets in that dataset are not in the PDF, so this repo ships with a freshly written 30-tweet dataset of the same size and structure. Your numbers on screen will therefore differ from the report's tables. Before you submit or present, either (a) replace the numbers in Section 7 of the report with the ones from `reports/accuracy_report.txt` and `python predict_sample.py`, or (b) put your original `tweets.csv` into `dataset/` and re-run `python train_model.py`. Don't present numbers that the live demo doesn't reproduce.

---

## 7. Likely viva questions

**Why TF-IDF and not simple word counts?**
It down-weights words that appear in almost every tweet and up-weights distinctive ones, which is the core idea of Information Retrieval ranking.

**Why Logistic Regression?**
Fast, lightweight, works well on sparse TF-IDF vectors, and its weights are interpretable.

**Why is the accuracy low?**
Only 30 tweets (6 in the test set). Many words in new tweets are unseen by the vectoriser and are ignored, so the model has little to go on.

**What preprocessing do you do and why?**
Lower-case (so "Happy" = "happy"), remove URLs, mentions, `#`, digits, punctuation (noise), tokenise, drop stop-words (words with no sentiment). The same `clean_text()` is used in training and prediction, which keeps them consistent.

**What does the system do when nothing is left after cleaning?**
For example a tweet with only a link: there are no sentiment-bearing words, so it returns Neutral.

**What are the limitations?**
Tiny dataset, bag-of-words ignores order so negation ("not good"), sarcasm, slang and emojis are not understood, English only, one tweet at a time.

**How is this related to Information Retrieval?**
Text preprocessing, tokenisation, stop-word removal and TF-IDF term weighting are standard IR techniques; here they feed a classifier instead of a ranking function.

**What are precision, recall and F1?**
Precision = correct predictions of a class ÷ all predictions of that class. Recall = correct predictions of a class ÷ all real tweets of that class. F1 = harmonic mean of the two.

---

## 8. Project structure

```
tweet-sentiment-analysis/
├── app.py                    # Flask web app (/, /predict, /about)
├── train_model.py            # load → clean → TF-IDF → split → train → evaluate → save
├── predict_sample.py         # batch prediction on dataset/sample_test.csv
├── config.py                 # paths and constants
├── setup_nltk.py             # optional NLTK data download
├── requirements.txt
├── dataset/
│   ├── tweets.csv            # 30 labelled tweets (10 / 10 / 10)
│   ├── sample_test.csv       # 15 unlabelled demo tweets
│   └── sample_test_expected.csv  # manual labels used by predict_sample.py
├── preprocessing/
│   ├── clean_text.py         # cleaning pipeline
│   ├── tokenizer.py          # NLTK tokenisation
│   └── stopwords.py          # NLTK English stop-words
├── utils/
│   ├── predictor.py          # loads .pkl files, predicts a tweet
│   └── metrics.py            # accuracy, precision, recall, F1, confusion matrix
├── model/                    # sentiment_model.pkl, tfidf_vectorizer.pkl
├── reports/                  # accuracy_report.txt, classification_report.txt, confusion_matrix.png
├── templates/                # index.html, result.html, about.html
├── static/                   # css/style.css, js/script.js
└── tests/                    # pytest cases
```

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| Port 5000 already in use | Close the other app, or change the port in the last line of `app.py`: `app.run(debug=True, port=5001)` |
| scikit-learn version warning when loading the model | Run `python train_model.py` once to regenerate the `.pkl` files |
| NLTK download fails (no internet) | Ignore it, the built-in fallback gives the same results |
| Emojis show as boxes | Use Chrome / Edge / Firefox |

## 10. Push to GitHub

```bash
git init
git add .
git commit -m "Tweet Sentiment Analysis System"
git branch -M main
git remote add origin https://github.com/<your-username>/tweet-sentiment-analysis.git
git push -u origin main
```

## Team

| Name | Roll No. | PRN |
|---|---|---|
| Ahmed Tabish | 45 | F23113054 |
| Abhishek Chavhan | 46 | F23113055 |
| Rohan Pardhi | 47 | F23113056 |
| Shejal Sonawane | 48 | F23113057 |
