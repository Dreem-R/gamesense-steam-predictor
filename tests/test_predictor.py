import json
import os
from pathlib import Path

import numpy as np
import pytest

from gamesense import config
from gamesense.game import Game
from gamesense.presets import PRESETS

SNAPSHOT = Path(__file__).parent / "data" / "preset_predictions.json"


def test_prediction_is_well_formed(predictor, preset_game):
    pred = predictor.predict(preset_game)
    assert pred.proba.shape == (4,)
    assert pred.proba.sum() == pytest.approx(1)
    assert (pred.proba >= 0).all()
    assert pred.tier == int(np.argmax(pred.proba))
    assert pred.tier_name == config.TIER_NAMES[pred.tier]
    assert pred.success_chance == pytest.approx(pred.proba[2:].sum())
    assert 0 <= pred.reviews_low <= pred.reviews_mid <= pred.reviews_high


def test_text_model_needs_a_description(predictor):
    assert not predictor.predict(Game(about="Short.")).used_text
    assert predictor.predict(PRESETS["Story-driven visual novel"]).used_text


def test_predictions_match_snapshot(predictor):
    """Catches unintended changes to features or the model. After retraining, regenerate
    with UPDATE_SNAPSHOT=1 pytest tests/test_predictor.py."""
    current = {}
    for name, game in PRESETS.items():
        pred = predictor.predict(game)
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
    first_game = PRESETS["Story-driven visual novel"]
    proven_studio = first_game.with_changes(dev_prior_games=3, dev_prior_best_reviews=20_000)
    assert predictor.predict(proven_studio).success_chance > predictor.predict(first_game).success_chance


def test_explain_returns_top_contributions(predictor, preset_game):
    contributions = predictor.explain(preset_game, top_k=6)
    features = [c.feature for c in contributions]
    effects = [abs(c.effect) for c in contributions]
    assert len(contributions) == 6
    assert "year" not in features
    assert not any(f.startswith("genre_") for f in features)
    assert effects == sorted(effects, reverse=True)
    for c in contributions:
        assert c.multiplier == pytest.approx(np.exp(c.effect))


def test_what_if_scores_each_change(predictor):
    game = PRESETS["Solo developer's first platformer"]
    base = predictor.predict(game)
    scored, plan = predictor.what_if(game)

    assert len(scored) == len({s.change.id for s in scored}) > 0
    assert all(s.success_before == base.success_chance for s in scored)
    uplifts = [s.uplift for s in scored]
    assert uplifts == sorted(uplifts, reverse=True)

    single = next(s for s in scored if s.change.id == "steam_cloud")
    assert single.success_after == pytest.approx(predictor.predict(single.change.apply(game)).success_chance)

    cheap_helping = [s.change.id for s in scored if s.helps and s.change.effort != "High"]
    if len(cheap_helping) > 1:
        assert plan is not None
        assert set(plan.change_ids) == set(cheap_helping)
        assert plan.tier_after in config.TIER_NAMES
    else:
        assert plan is None


def test_what_if_with_nothing_to_change(predictor):
    complete = Game(
        n_screenshots=12,
        has_website=True,
        achievements=30,
        n_languages=10,
        linux=True,
        mac=True,
        price=19.99,
        self_published=False,
        categories=["Single-player", "Steam Cloud", "Full controller support", "Co-op"],
    )
    assert predictor.what_if(complete) == ([], None)
