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

st.set_page_config(page_title="GameSense · Will my game succeed?", page_icon="🎮", layout="wide")

pages = [
    st.Page("views/predict.py", title="Predict my game", icon="🎮", default=True),
    st.Page("views/insights.py", title="Market insights", icon="📊"),
    st.Page("views/how_it_works.py", title="How it works", icon="🧠"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🎮 GameSense")
    st.caption("Predicts how a game will perform on Steam from its store page and studio track record. "
               "Machine-learning project trained on 69k real Steam releases.")

nav.run()
