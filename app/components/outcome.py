"""The headline prediction: tier card, key numbers and tier probabilities."""

import plotly.graph_objects as go
import streamlit as st

from common import TIER_COLOR, TIER_HEADLINE
from gamesense import config
from gamesense.game import Game
from gamesense.predictor import Prediction
from gamesense.reliability import confidence_level


def tier_card(game: Game, pred: Prediction) -> None:
    tier = pred.tier_name
    color = TIER_COLOR[tier]
    level, _ = confidence_level(pred.proba)
    st.markdown(
        f"""<div style="border-radius:10px;padding:16px 20px;background:{color}1f;
        border:1px solid {color};margin-bottom:12px">
        <div style="font-size:0.85rem;opacity:.75">Prediction for {game.name or "your game"}</div>
        <div style="font-size:2rem;font-weight:700;line-height:1.25">{tier}</div>
        <div style="font-size:1.05rem;font-weight:600">{TIER_HEADLINE[tier]}</div>
        <div style="font-size:0.85rem;opacity:.8;margin-top:4px">{config.TIER_BLURBS[tier]}
        Confidence: {level.lower()}.</div></div>""",
        unsafe_allow_html=True,
    )


def render(game: Game, pred: Prediction, market_rate: float) -> None:
    tier_card(game, pred)

    m1, m2 = st.columns(2)
    ratio = pred.success_chance / market_rate
    m1.metric(
        "Chance of 100+ reviews",
        f"{pred.success_chance:.0%}",
        f"{ratio:.1f}x the {config.YEAR_MAX} average",
        delta_color="normal" if ratio >= 1 else "inverse",
    )
    m2.metric(
        "Expected reviews",
        f"{pred.reviews_mid:,.0f}",
        f"likely range {pred.reviews_low:,.0f} to {pred.reviews_high:,.0f}",
        delta_color="off",
    )
    st.caption(
        f"At 30 to 60 sales per review, that is roughly {pred.reviews_mid * 30:,.0f} to "
        f"{pred.reviews_mid * 60:,.0f} copies."
    )

    fig = go.Figure(
        go.Bar(
            x=pred.proba * 100,
            y=config.TIER_NAMES,
            orientation="h",
            marker_color=[TIER_COLOR[t] for t in config.TIER_NAMES],
            text=[f"{p:.0%}" for p in pred.proba],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=30, t=30, b=0),
        title="Probability of each outcome",
        xaxis=dict(range=[0, 108], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
