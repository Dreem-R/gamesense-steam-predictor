"""GameSense Streamlit app.

Usage (from the project root): streamlit run app/streamlit_app.py
"""
# lightgbm must be imported before streamlit (which loads pyarrow), otherwise
# prediction segfaults on Windows.
import lightgbm  # noqa: F401, I001

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="GameSense | Steam success predictor", layout="wide")

pages = [
    st.Page("views/predict.py", title="Predict", default=True),
    st.Page("views/insights.py", title="Market insights"),
    st.Page("views/how_it_works.py", title="How it works"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### GameSense")
    st.caption("Estimates how a game will perform on Steam from its store page and the studio's track record. "
               "Trained on 69,127 Steam releases from 2018 to 2024.")

nav.run()
