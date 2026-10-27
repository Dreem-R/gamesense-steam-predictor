"""Chart of the features pushing the expected review count up or down."""

import plotly.graph_objects as go
import streamlit as st

from common import TIER_COLOR
from gamesense.predictor import Contribution


def render(contributions: list[Contribution]) -> None:
    st.subheader("What is driving the prediction")
    rows = contributions[::-1]
    fig = go.Figure(
        go.Bar(
            x=[c.effect for c in rows],
            y=[f"{c.label} ({c.value})" for c in rows],
            orientation="h",
            marker_color=[TIER_COLOR["Success"] if c.effect > 0 else TIER_COLOR["Flop"] for c in rows],
            text=[f"x{c.multiplier:.2f}" for c in rows],
            textposition="auto",
            hovertemplate="%{y}<br>reviews %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="lowers expected reviews  |  raises expected reviews",
        xaxis_zeroline=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Each bar shows how much a factor moves the expected review count compared with an average game "
        "(SHAP values from the review model). x1.50 means 50% more reviews, x0.70 means 30% fewer."
    )
