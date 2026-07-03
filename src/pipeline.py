"""Distributed preprocessing: raw tweets to a cleaned, sentiment-labeled DataFrame."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType
from textblob import TextBlob

from src.text_cleaning import (
    NON_ALPHANUMERIC_PATTERN,
    RETWEET_PREFIX_PATTERN,
    URL_PATTERN,
    categorize_sentiment,
    remove_stopwords,
)


def clean_tweets(df, text_column="text"):
    """Clean a DataFrame of raw tweets.

    Mirrors the cleaning chain from the original EMR notebook: drop nulls and
    duplicates, strip URLs, line breaks and retweet prefixes, de-duplicate
    again (retweets collapse onto the original tweet once the prefix is gone),
    normalize case and punctuation, and remove stop words.
    """
    remove_stopwords_udf = F.udf(remove_stopwords, StringType())

    return (
        df.select(F.col(text_column).alias("tweets"))
        .na.drop()
        .dropDuplicates()
        .withColumn("tweets", F.regexp_replace("tweets", URL_PATTERN, ""))
        .withColumn("tweets", F.regexp_replace("tweets", r"\n", " "))
        .withColumn("tweets", F.regexp_replace("tweets", RETWEET_PREFIX_PATTERN, ""))
        .dropDuplicates()
        .withColumn("tweets", F.lower("tweets"))
        .withColumn("tweets", F.regexp_replace("tweets", NON_ALPHANUMERIC_PATTERN, " "))
        .withColumn("tweets", remove_stopwords_udf("tweets"))
    )


def label_sentiment(df, text_column="tweets"):
    """Add TextBlob polarity scores and bucketed sentiment categories.

    The corpus has no ground-truth labels, so TextBlob acts as a weak
    supervision source; the classifier in ``train_classifier`` learns to
    reproduce these labels with native Spark ML.
    """
    polarity_udf = F.udf(lambda text: TextBlob(text).sentiment.polarity, FloatType())
    category_udf = F.udf(categorize_sentiment, StringType())

    return (
        df.withColumn("sentiment_score", polarity_udf(text_column))
        .withColumn("sentiment_category", category_udf("sentiment_score"))
    )
