"""Reference data shipped with the repository: market sample, held-out predictions and metrics."""

import json
from typing import Any

import pandas as pd

from gamesense import config


def load_market() -> pd.DataFrame:
    """Reduced table of all cleaned games, used for market statistics and input warnings."""
    df = pd.read_parquet(config.APP_DATA_PATH)
    df["genres"] = df["genres"].str.split("|")
    df["categories"] = df["categories"].str.split("|")
    df["success"] = df["tier"] >= 2
    return df


def load_test_predictions() -> pd.DataFrame:
    """Predictions for the held-out test year, used for per-prediction reliability checks."""
    return pd.read_parquet(config.TEST_PREDICTIONS_PATH)


def load_metrics() -> dict[str, Any]:
    return json.loads(config.METRICS_PATH.read_text())
