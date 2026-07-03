"""Train and evaluate a TF-IDF + Naive Bayes sentiment classifier with Spark ML."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, StringIndexer, Tokenizer

METRICS = ("accuracy", "weightedPrecision", "weightedRecall", "f1")


def build_pipeline(num_features=1000):
    """TF-IDF features into multinomial Naive Bayes, as a single ML Pipeline.

    Wrapping the stages in a Pipeline (rather than fitting them one by one)
    keeps the IDF model fitted on training data only, avoiding the subtle
    train/test leakage present in the original notebook.
    """
    return Pipeline(stages=[
        StringIndexer(inputCol="sentiment_category", outputCol="label"),
        Tokenizer(inputCol="tweets", outputCol="words"),
        HashingTF(inputCol="words", outputCol="raw_features", numFeatures=num_features),
        IDF(inputCol="raw_features", outputCol="features"),
        NaiveBayes(featuresCol="features", labelCol="label"),
    ])


def train_and_evaluate(df, train_ratio=0.7, num_features=1000, seed=42):
    """Split, train, and score. Returns the fitted model and a metrics dict."""
    train, test = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    model = build_pipeline(num_features).fit(train)
    predictions = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                  predictionCol="prediction")
    metrics = {
        name: evaluator.evaluate(predictions, {evaluator.metricName: name})
        for name in METRICS
    }
    return model, metrics
