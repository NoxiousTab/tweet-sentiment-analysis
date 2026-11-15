"""Tweet cleaning: lower-case, strip noise, tokenise, remove stop-words."""
import re
import string

from .stopwords import STOP_WORDS
from .tokenizer import tokenize_text


def clean_text(text):
    """Return the cleaned version of a raw tweet as a single string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"#", "", text)  # hashtag symbol (the word is kept)
    text = re.sub(r"\d+", "", text)  # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in tokenize_text(text) if w not in STOP_WORDS]
    return " ".join(words)
