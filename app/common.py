"""Cached loaders and display constants."""

from typing import Any

import pandas as pd
import streamlit as st

from gamesense import config, datasets
from gamesense.predictor import GameSensePredictor

# Used by the model but not offered as choices in the app.
HIDDEN_GENRES = {"Nudity", "Sexual Content"}
VISIBLE_GENRES = [g for g in config.GENRES if g not in HIDDEN_GENRES]

TEST_YEAR = config.TEST_YEARS[0]
TIER_COLOR = {"Flop": "#d1495b", "Niche": "#edae49", "Success": "#66a182", "Hit": "#2e86ab"}
TIER_HEADLINE = {
    "Flop": "Likely to struggle to find players",
    "Niche": "Likely to find a small audience",
    "Success": "Likely to find a solid audience",
    "Hit": "Likely to reach a large audience",
}


def md(text: str) -> str:
    """Escape dollar signs so Streamlit does not render them as LaTeX."""
    return text.replace("$", r"\$")


@st.cache_resource(show_spinner="Loading the model...")
def load_predictor() -> GameSensePredictor:
    return GameSensePredictor()


@st.cache_data
def load_insights() -> pd.DataFrame:
    return datasets.load_market()


@st.cache_data
def load_metrics() -> dict[str, Any]:
    return datasets.load_metrics()


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return datasets.load_test_predictions()
