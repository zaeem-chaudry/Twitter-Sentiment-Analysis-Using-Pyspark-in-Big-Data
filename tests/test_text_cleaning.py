import re

from src.text_cleaning import (
    RETWEET_PREFIX_PATTERN,
    categorize_sentiment,
    clean_text,
    remove_stopwords,
)


class TestCleanText:
    def test_urls_are_removed(self):
        assert "https" not in clean_text("Vaccine rollout update https://t.co/abc123")

    def test_line_breaks_become_spaces(self):
        assert "\n" not in clean_text("first line\nsecond line")

    def test_retweet_prefix_is_stripped(self):
        assert clean_text("RT @WHO: Stay safe").startswith("stay safe")

    def test_retweet_marker_mid_text_is_kept(self):
        # only the leading "RT @user: " prefix is a retweet marker
        assert re.sub(RETWEET_PREFIX_PATTERN, "", "thanks RT @WHO: ok") == "thanks RT @WHO: ok"

    def test_lowercases_and_replaces_punctuation(self):
        assert clean_text("COVID-19 cases RISING!").split() == ["covid", "19", "cases", "rising"]

    def test_full_chain(self):
        raw = "RT @WHO: New COVID-19 guidance!\nDetails: https://t.co/xyz"
        assert remove_stopwords(clean_text(raw)) == "new covid 19 guidance details"


class TestRemoveStopwords:
    def test_drops_stop_words(self):
        assert remove_stopwords("the who is monitoring the outbreak") == "who monitoring outbreak"

    def test_is_case_insensitive(self):
        assert remove_stopwords("The WHO IS here") == "WHO here"

    def test_empty_string(self):
        assert remove_stopwords("") == ""


class TestCategorizeSentiment:
    def test_positive(self):
        assert categorize_sentiment(0.8) == "Positive"

    def test_negative(self):
        assert categorize_sentiment(-0.8) == "Negative"

    def test_neutral(self):
        assert categorize_sentiment(0.0) == "Neutral"

    def test_thresholds_are_exclusive(self):
        assert categorize_sentiment(0.5) == "Neutral"
        assert categorize_sentiment(-0.5) == "Neutral"

    def test_custom_thresholds(self):
        assert categorize_sentiment(0.2, positive_threshold=0.1) == "Positive"
        assert categorize_sentiment(-0.2, negative_threshold=-0.1) == "Negative"
