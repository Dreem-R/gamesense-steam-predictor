"""Cached loaders and display constants."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predictor import GameSensePredictor  # noqa: E402

# Used by the model but not offered as choices in the app.
HIDDEN_GENRES = {"Nudity", "Sexual Content"}
VISIBLE_GENRES = [g for g in config.GENRES if g not in HIDDEN_GENRES]

TIER_COLOR = {"Flop": "#d1495b", "Niche": "#edae49", "Success": "#66a182", "Hit": "#2e86ab"}
TIER_HEADLINE = {
    "Flop": "Likely to struggle to find players",
    "Niche": "Likely to find a small audience",
    "Success": "Likely to find a solid audience",
    "Hit": "Likely to reach a large audience",
}


@st.cache_resource(show_spinner="Loading the model...")
def load_predictor() -> GameSensePredictor:
    return GameSensePredictor()


@st.cache_data
def load_insights() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "app" / "data" / "insights.parquet")
    df["genres"] = df["genres"].str.split("|")
    df["categories"] = df["categories"].str.split("|")
    df["success"] = df["tier"] >= 2
    return df


@st.cache_data
def load_metrics() -> dict:
    return json.loads((config.MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.REPORTS_DIR / "test2024_predictions.parquet")
