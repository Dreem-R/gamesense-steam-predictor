import streamlit as st

from common import load_insights, load_metrics, load_predictor, load_test_predictions
from components import drivers, form, outcome, reliability, suggestions
from gamesense import config

predictor = load_predictor()
form.init_state()

st.title("Will my game succeed on Steam?")
st.markdown(
    f"Describe the game you are making. The model was trained on **{predictor.model.train_rows:,} Steam games "
    f"released between {config.YEAR_MIN} and {config.YEAR_MAX}** and estimates how a game like yours performs "
    "in its first year or two on sale."
)
form.preset_picker()

left, right = st.columns([1.05, 1], gap="large")
with left:
    form.render()

game = form.current_game()
market = load_insights()
with right:
    if game is None:
        st.info("Select at least one genre to get a prediction.")
        st.stop()
    pred = predictor.predict(game)
    outcome.render(game, pred, market_rate=market.loc[market.year == config.YEAR_MAX, "success"].mean())

st.divider()
reliability.render(game, pred, load_test_predictions(), load_metrics(), market)
st.divider()
drivers.render(predictor.explain(game, top_k=8))
st.divider()
suggestions.render(game, pred, *predictor.what_if(game))
