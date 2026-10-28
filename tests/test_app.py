from pathlib import Path
from typing import get_args

import pytest
from streamlit.testing.v1 import AppTest

from components.suggestions import TEXT, title
from gamesense.features import slug
from gamesense.game import Game
from gamesense.presets import PRESETS
from gamesense.suggestions import ChangeId, candidate_changes

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def run_app(page: str | None = None, **query: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params.update(query)
    if page:
        at.switch_page(page)
    return at.run()


@pytest.mark.parametrize(
    ("page", "title"),
    [
        (None, "Will my game succeed on Steam?"),
        ("views/insights.py", "Steam market insights"),
        ("views/how_it_works.py", "How it works"),
        ("views/api.py", "API reference"),
    ],
)
def test_page_renders_without_errors(page, title):
    at = run_app(page)
    assert not at.exception
    assert at.title[0].value == title


def test_predict_page_shows_reliability_and_suggestions():
    at = run_app()
    subheaders = [s.value for s in at.subheader]
    assert "How reliable is this prediction?" in subheaders
    assert "Suggestions" in subheaders


def test_example_can_be_loaded_from_the_url():
    at = run_app(example="story-driven-visual-novel")
    assert at.text_input(key="g_name").value == "Letters to Autumn"


def test_prediction_requires_a_genre():
    at = run_app()
    at.multiselect(key="g_genres").set_value([]).run()
    assert not at.exception
    assert any("at least one genre" in i.value for i in at.info)


def test_multiplayer_only_example_has_no_single_player():
    at = run_app(example="free-to-play-online-shooter")
    assert at.checkbox(key="g_mp_only").value


@pytest.mark.parametrize("name", list(PRESETS))
def test_form_rebuilds_each_example(predictor, name):
    at = run_app(example=slug(name).replace("_", "-"))
    pred = predictor.predict(PRESETS[name])
    shown = {m.label: m.value for m in reversed(at.metric)}
    assert shown["Chance of 100+ reviews"] == f"{pred.success_chance:.0%}"
    assert shown["Expected reviews"] == f"{pred.reviews_mid:,.0f}"


def test_api_reference_embeds_the_api_spec():
    at = run_app("views/api.py")
    html = at.get("iframe")[0].proto.srcdoc
    assert '"title": "GameSense API"' in html
    assert "http://localhost:8000/docs" in html


def test_every_suggestion_has_text():
    assert TEXT.keys() == set(get_args(ChangeId))


def test_price_suggestion_title_uses_both_prices():
    change = next(c for c in candidate_changes(Game(price=2.99)) if c.id == "raise_price")
    assert title(change, Game(price=2.99)) == "Price at $14.99 instead of $2.99"
