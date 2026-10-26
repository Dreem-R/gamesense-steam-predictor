import json
import os
from pathlib import Path

import numpy as np
import pytest

from gamesense import config
from gamesense.predictor import SUGGESTION_COLS, candidate_changes, default_game
from gamesense.presets import PRESETS

SNAPSHOT = Path(__file__).parent / "data" / "preset_predictions.json"


def test_prediction_is_well_formed(predictor, preset_game):
    pred = predictor.predict(preset_game)
    assert pred.proba.shape == (4,)
    assert pred.proba.sum() == pytest.approx(1)
    assert (pred.proba >= 0).all()
    assert pred.tier == int(np.argmax(pred.proba))
    assert pred.success_chance == pytest.approx(pred.proba[2:].sum())
    assert 0 <= pred.reviews_low <= pred.reviews_mid <= pred.reviews_high


def test_text_model_needs_a_description(predictor):
    assert not predictor.predict(default_game(about="Short.")).used_text
    assert predictor.predict(default_game(**PRESETS["Story-driven visual novel"])).used_text


def test_predictions_match_snapshot(predictor):
    """Catches unintended changes to features or the model. After retraining, regenerate
    with UPDATE_SNAPSHOT=1 pytest tests/test_predictor.py."""
    current = {}
    for name, overrides in PRESETS.items():
        pred = predictor.predict(default_game(**overrides))
        current[name] = {"proba": pred.proba.round(6).tolist(), "reviews_mid": round(pred.reviews_mid, 3)}

    if os.environ.get("UPDATE_SNAPSHOT"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2) + "\n")

    expected = json.loads(SNAPSHOT.read_text())
    assert current.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(current[name]["proba"], expected[name]["proba"], atol=1e-4, err_msg=name)
        assert current[name]["reviews_mid"] == pytest.approx(expected[name]["reviews_mid"], rel=1e-3), name


def test_studio_track_record_raises_success_chance(predictor):
    first_game = default_game(**PRESETS["Story-driven visual novel"])
    proven_studio = dict(first_game, dev_prior_games=3, dev_prior_best_reviews=20_000)
    assert predictor.predict(proven_studio).success_chance > predictor.predict(first_game).success_chance


def test_explain_returns_top_contributions(predictor, preset_game):
    exp = predictor.explain(preset_game, top_k=6)
    assert len(exp) == 6
    assert "year" not in set(exp.feature)
    assert not exp.feature.str.startswith("genre_").any()
    assert exp.effect.abs().is_monotonic_decreasing
    np.testing.assert_allclose(exp.multiplier, np.exp(exp.effect))


def test_what_if_scores_each_change(predictor):
    game = default_game(**PRESETS["Solo developer's first platformer"])
    base = predictor.predict(game)
    table, plan = predictor.what_if(game)

    assert list(table.columns) == SUGGESTION_COLS
    assert len(table) == len(candidate_changes(game))
    assert (table.success_before == base.success_chance).all()
    assert table.uplift.is_monotonic_decreasing
    np.testing.assert_allclose(table.uplift, table.success_after - table.success_before)

    helpful_cheap = set(table.loc[table.helps & (table.effort != "High"), "change"])
    if len(helpful_cheap) > 1:
        assert set(plan["changes"]) == helpful_cheap
        assert plan["tier_after"] in config.TIER_NAMES
    else:
        assert plan == {}


def test_what_if_with_nothing_to_change(predictor):
    complete = default_game(
        n_screenshots=12,
        has_website=1,
        achievements=30,
        n_languages=10,
        linux=1,
        mac=1,
        price=19.99,
        self_published=0,
        categories=["Single-player", "Steam Cloud", "Full controller support", "Co-op"],
    )
    table, plan = predictor.what_if(complete)
    assert table.empty
    assert list(table.columns) == SUGGESTION_COLS
    assert plan == {}


def changes_for(**overrides) -> dict[str, dict]:
    return {o["change"]: o for o in candidate_changes(default_game(**overrides))}


def test_cheap_games_get_a_price_suggestion():
    options = changes_for(price=2.99)
    assert options["Price at $14.99 instead of $2.99"]["changes"] == {"price": 14.99}


def test_free_games_get_no_price_suggestion():
    assert not any(c.startswith("Price at") for c in changes_for(price=0.0))


def test_suggestions_add_categories_without_dropping_existing_ones():
    options = changes_for(categories=["Single-player", "Steam Workshop"])
    cats = options["Enable Steam Cloud saves"]["changes"]["categories"]
    assert set(cats) == {"Single-player", "Steam Workshop", "Steam Cloud"}


def test_publisher_suggestion_is_high_effort_and_only_for_self_published():
    assert changes_for(self_published=1)["Work with an established publisher"]["effort"] == "High"
    assert "Work with an established publisher" not in changes_for(self_published=0)


def test_default_game_counts_description_words():
    assert default_game(about="one two  three")["about_words"] == 3
    assert default_game(about="one two", about_words=99)["about_words"] == 99
