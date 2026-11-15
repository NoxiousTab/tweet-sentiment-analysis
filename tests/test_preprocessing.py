from preprocessing import clean_text


def test_lowercase():
    assert clean_text("HAPPY Day") == "happy day"


def test_removes_urls_mentions_hashtag_symbol_and_digits():
    cleaned = clean_text("Check this https://t.co/xyz @friend #awesome 123")
    assert "http" not in cleaned
    assert "@" not in cleaned and "#" not in cleaned
    assert not any(ch.isdigit() for ch in cleaned)
    assert "awesome" in cleaned
    assert "friend" not in cleaned


def test_removes_punctuation():
    assert clean_text("Wow!!! Great, really.") == "wow great really"


def test_removes_stopwords():
    cleaned = clean_text("this is the best")
    assert cleaned == "best"


def test_non_string_input_is_safe():
    assert clean_text(None) == ""
