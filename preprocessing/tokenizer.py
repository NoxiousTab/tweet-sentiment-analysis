"""Word tokenisation using NLTK.

Order of preference:
1. nltk.word_tokenize (needs the 'punkt' data)
2. nltk.wordpunct_tokenize (pure NLTK regex tokenizer, needs no data)
3. str.split (only if NLTK itself is not installed)
"""


def _build_tokenizer():
    try:
        from nltk.tokenize import word_tokenize

        word_tokenize("test")  # raises LookupError if punkt data is missing
        return word_tokenize
    except Exception:
        pass
    try:
        from nltk.tokenize import wordpunct_tokenize

        return wordpunct_tokenize
    except Exception:
        return str.split


_tokenizer = _build_tokenizer()


def tokenize_text(text):
    """Split a (already cleaned) string into a list of word tokens."""
    return _tokenizer(text)
