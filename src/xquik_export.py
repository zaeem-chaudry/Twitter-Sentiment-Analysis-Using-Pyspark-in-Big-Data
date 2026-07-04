"""Helpers for using Xquik tweet exports with the Spark pipeline.

The main pipeline expects a tweet text column named ``text``. Xquik exports can
arrive with source-specific names, so this module normalizes those files before
they are passed to ``src.run_pipeline``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

TEXT_COLUMN_CANDIDATES = ("text", "tweet", "tweet_text", "full_text", "content")


def find_text_column(columns: Iterable[str], preferred: str | None = None) -> str:
    """Return the first usable tweet-text column name.

    Args:
        columns: DataFrame column names.
        preferred: Optional caller-selected text column.

    Raises:
        ValueError: If no supported text column exists.
    """
    available = {column.lower(): column for column in columns}
    if preferred:
        preferred_key = preferred.lower()
        if preferred_key not in available:
            raise ValueError(f"Preferred text column not found: {preferred}")
        return available[preferred_key]

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in available:
            return available[candidate]

    supported = ", ".join(TEXT_COLUMN_CANDIDATES)
    raise ValueError(f"Xquik export needs one of these text columns: {supported}")


def normalize_xquik_export(df: pd.DataFrame, text_column: str | None = None) -> pd.DataFrame:
    """Normalize an Xquik export DataFrame for ``src.run_pipeline``.

    The returned DataFrame always has a non-empty ``text`` column. Empty rows are
    dropped so Spark does not spend work on unusable records.
    """
    source_column = find_text_column(df.columns, preferred=text_column)
    normalized = df.copy()

    if source_column != "text":
        normalized = normalized.rename(columns={source_column: "text"})

    normalized["text"] = normalized["text"].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["text"] != ""].reset_index(drop=True)

    if "source" not in normalized.columns:
        normalized["source"] = "xquik"

    return normalized
