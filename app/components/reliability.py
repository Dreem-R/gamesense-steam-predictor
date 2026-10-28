"""How far to trust a prediction, based on held-out results."""

from typing import Any

import pandas as pd
import streamlit as st

from common import TEST_YEAR, md
from gamesense.game import Game
from gamesense.predictor import Prediction
from gamesense.reliability import (
    MIN_CALIBRATION_GAMES,
    calibration_check,
    confidence_level,
    input_warnings,
    tier_track_record,
)


def render(game: Game, pred: Prediction, test: pd.DataFrame, metrics: dict[str, Any], market: pd.DataFrame) -> None:
    st.subheader("How reliable is this prediction?")
    st.markdown(
        "These checks compare this prediction with how the model performed on "
        f"**{metrics['rows_test']:,} games released in {TEST_YEAR}**, which were held out from training."
    )

    tier = pred.tier_name
    level, level_detail = confidence_level(pred.proba)
    track = tier_track_record(test, pred.tier)
    calib = calibration_check(test, pred.success_chance)
    r1, r2, r3 = st.columns(3)
    with r1, st.container(border=True):
        st.metric("Confidence", level)
        st.caption(
            md(level_detail) + " When probabilities are spread across tiers, treat the tier label "
            "as a rough guide and look at the chance of 100+ reviews instead."
        )
    with r2, st.container(border=True):
        st.metric(f"Past accuracy for {tier} predictions", f"{track['exact']:.0%}")
        st.caption(
            f"Of {track['games']:,} games from {TEST_YEAR} that the model placed in {tier}, "
            f"{track['exact']:.0%} ended up there and {track['within_one']:.0%} were within one tier."
        )
    with r3, st.container(border=True):
        if calib["games"] >= MIN_CALIBRATION_GAMES:
            st.metric("Did similar predictions come true?", f"{calib['actual_rate']:.0%}")
            st.caption(
                f"{calib['games']:,} games from {TEST_YEAR} were given a {calib['low']:.0%} to "
                f"{calib['high']:.0%} chance of 100+ reviews. {calib['actual_rate']:.0%} of them "
                "actually got there."
            )
        else:
            st.metric("Did similar predictions come true?", "Too few cases")
            st.caption(
                f"Fewer than {MIN_CALIBRATION_GAMES} games from {TEST_YEAR} were given a chance this close, "
                "so there is not enough data to check it."
            )

    coverage = metrics["review_range_test_2024"]["interval_80_coverage"]
    st.caption(
        f"The review range is the model's 10th to 90th percentile estimate. For {TEST_YEAR} games, the actual "
        f"review count fell inside that range {coverage:.0%} of the time, so the real number can land outside it."
    )

    for note in input_warnings(game, market):
        st.warning(md(note))

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**What the model uses**")
        st.markdown(
            "- Genres, price, platforms and Steam features\n"
            "- Store page setup: screenshots, languages, achievements, description\n"
            "- The studio's and publisher's previous releases\n"
            f"- Patterns from {metrics['rows_total']:,} real launches"
        )
    with k2:
        st.markdown("**What the model cannot see**")
        st.markdown(
            "- Whether the game is fun, polished or original\n"
            "- Trailer quality, capsule art and wishlists before launch\n"
            "- Marketing, press, festivals and streamer coverage\n"
            "- Launch timing against big releases and Steam sales"
        )
    st.info(
        "Best use: change one thing at a time and compare the results, instead of treating a single number "
        "as a forecast. A low score does not mean the idea is bad. It means games with a similar store setup "
        "usually struggled, and the parts the model cannot see need to do more of the work."
    )
