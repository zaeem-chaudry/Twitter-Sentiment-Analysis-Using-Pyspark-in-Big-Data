"""Pure text-cleaning rules shared by the Spark pipeline and the unit tests.

This module has no PySpark dependency, so the core cleaning logic can be
unit-tested without a running Spark cluster. The regex patterns are kept as
plain strings because they are passed both to Python's ``re`` module and to
Spark's ``regexp_replace`` (Java regex) — they are valid in both dialects.
"""

import re

URL_PATTERN = r"http\S+"
RETWEET_PREFIX_PATTERN = r"^RT @\w+: "
NON_ALPHANUMERIC_PATTERN = r"[^a-zA-Z0-9\s]"

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "is", "it", "of", "on", "in",
    "to", "for", "with", "as", "at", "by", "from", "that", "this", "these",
    "those", "then", "there", "when", "where", "how", "why",
})

POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


def clean_text(text):
    """Apply the cleaning chain to a single tweet.

    Order matters: the retweet prefix is case-sensitive, so it is stripped
    before lowercasing, and URLs are removed before punctuation is replaced.
    """
    text = re.sub(URL_PATTERN, "", text)
    text = text.replace("\n", " ")
    text = re.sub(RETWEET_PREFIX_PATTERN, "", text)
    text = text.lower()
    text = re.sub(NON_ALPHANUMERIC_PATTERN, " ", text)
    return text


def remove_stopwords(text):
    """Drop stop words and collapse whitespace."""
    words = [word for word in text.split() if word.lower() not in STOP_WORDS]
    return " ".join(words)


def categorize_sentiment(polarity,
                         positive_threshold=POSITIVE_THRESHOLD,
                         negative_threshold=NEGATIVE_THRESHOLD):
    """Bucket a TextBlob polarity score in [-1, 1] into a sentiment class."""
    if polarity > positive_threshold:
        return "Positive"
    if polarity < negative_threshold:
        return "Negative"
    return "Neutral"
