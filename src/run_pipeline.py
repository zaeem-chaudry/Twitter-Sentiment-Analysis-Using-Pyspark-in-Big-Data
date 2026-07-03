"""Command-line entry point for the sentiment analysis pipeline.

Run from the repository root so the ``src`` package resolves:

    python -m src.run_pipeline --input path/to/tweets.csv --text-column text \
        --output out/labeled --train

Works against any CSV with a tweet-text column. On a cluster, submit with
``spark-submit --py-files src`` instead of a local ``python -m`` run.
"""

import argparse

from pyspark.sql import SparkSession

from src.pipeline import clean_tweets, label_sentiment
from src.train_classifier import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Twitter sentiment analysis with PySpark")
    parser.add_argument("--input", required=True,
                        help="input CSV path or glob with a tweet text column")
    parser.add_argument("--text-column", default="text",
                        help="name of the column holding the tweet text (default: text)")
    parser.add_argument("--output",
                        help="optional output path for the labeled dataset (CSV)")
    parser.add_argument("--train", action="store_true",
                        help="also train and evaluate the TF-IDF + Naive Bayes classifier")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("Twitter Sentiment Analysis").getOrCreate()

    raw = spark.read.csv(args.input, header=True, inferSchema=True)
    labeled = label_sentiment(clean_tweets(raw, args.text_column))

    labeled.groupBy("sentiment_category").count().show()

    if args.output:
        labeled.write.mode("overwrite").csv(args.output, header=True)

    if args.train:
        _, metrics = train_and_evaluate(labeled)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
