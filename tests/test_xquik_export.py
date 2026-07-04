import unittest

import pandas as pd

from src.xquik_export import find_text_column, normalize_xquik_export


class XquikExportTests(unittest.TestCase):
    def test_find_text_column_prefers_explicit_name(self):
        self.assertEqual(find_text_column(["tweet_text", "body"], preferred="body"), "body")

    def test_normalizes_tweet_text_column_to_text(self):
        df = pd.DataFrame({"tweet_text": [" Hello world ", ""], "likes": [3, 0]})

        normalized = normalize_xquik_export(df)

        self.assertEqual(list(normalized.columns), ["text", "likes", "source"])
        self.assertEqual(normalized.loc[0, "text"], "Hello world")
        self.assertEqual(normalized.loc[0, "source"], "xquik")
        self.assertEqual(len(normalized), 1)

    def test_preserves_existing_text_column_and_source(self):
        df = pd.DataFrame({"text": ["A post"], "source": ["manual"]})

        normalized = normalize_xquik_export(df)

        self.assertEqual(normalized.loc[0, "text"], "A post")
        self.assertEqual(normalized.loc[0, "source"], "manual")

    def test_raises_for_missing_text_column(self):
        df = pd.DataFrame({"message": ["no supported column"]})

        with self.assertRaisesRegex(ValueError, "text columns"):
            normalize_xquik_export(df)


if __name__ == "__main__":
    unittest.main()
