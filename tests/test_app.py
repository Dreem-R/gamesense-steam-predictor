from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
