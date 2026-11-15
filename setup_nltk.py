"""Optional one-time helper: download the NLTK data used by the project.

The project still works without it (it falls back to NLTK's regex tokenizer and a
built-in copy of the NLTK English stop-word list), but running this once gives
you the exact NLTK resources mentioned in the report.

    python setup_nltk.py
"""
import nltk

for resource in ("punkt", "punkt_tab", "stopwords"):
    ok = nltk.download(resource, quiet=True)
    print(f"{resource:<10} -> {'OK' if ok else 'not available (fallback will be used)'}")
