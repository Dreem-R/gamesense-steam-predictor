import math

import numpy as np
import pandas as pd
import pytest

from gamesense.predictor import default_game
from gamesense.reliability import calibration_check, confidence_level, input_warnings, tier_track_record


@pytest.mark.parametrize(
    ("proba", "level"),
    [
        ([0.70, 0.20, 0.05, 0.05], "High"),
        ([0.50, 0.25, 0.15, 0.10], "Moderate"),
        ([0.40, 0.35, 0.15, 0.10], "Low"),
    ],
)
def test_confidence_level(proba, level):
    assert confidence_level(np.array(proba))[0] == level


def test_confidence_detail_names_top_two_tiers():
    _, detail = confidence_level(np.array([0.1, 0.4, 0.35, 0.15]))
    assert "Niche" in detail
    assert "Success" in detail


@pytest.fixture
def test_set() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tier": [0, 0, 1, 2, 2, 3],
            "pred_tier": [0, 1, 1, 2, 2, 2],
            "p_success": [0.05, 0.10, 0.12, 0.40, 0.45, 0.50],
            "p_hit": [0.00, 0.00, 0.01, 0.12, 0.10, 0.20],
        }
    )


def test_tier_track_record(test_set):
    record = tier_track_record(test_set, tier=2)
    assert record["games"] == 3
    assert record["exact"] == pytest.approx(2 / 3)
    assert record["within_one"] == 1
    assert record["actual_mix"] == pytest.approx([0, 0, 2 / 3, 1 / 3])


def test_calibration_check_uses_nearby_predictions(test_set):
    result = calibration_check(test_set, success_chance=0.55)
    assert result["games"] == 2
    assert result["actual_rate"] == 1


def test_calibration_check_with_no_similar_games(test_set):
    result = calibration_check(test_set, success_chance=0.95)
    assert result["games"] == 0
    assert math.isnan(result["actual_rate"])


@pytest.fixture
def market() -> pd.DataFrame:
    return pd.DataFrame({"genres": [["Indie", "Action"]] * 150 + [["Racing"]] * 5})


def describe(**overrides) -> dict:
    base = {"about": " ".join(["word"] * 50), "genres": ["Indie", "Action"]}
    return default_game(**{**base, **overrides})


def test_typical_game_has_no_warnings(market):
    assert input_warnings(describe(), market) == []


@pytest.mark.parametrize(
    ("overrides", "phrase"),
    [
        ({"genres": ["Racing"]}, "genre combination"),
        ({"about": ""}, "no real store description"),
        ({"price": 79.99}, "above $60"),
        ({"n_languages": 25}, "20 languages"),
        ({"achievements": 500}, "300 achievements"),
        ({"dev_prior_games": 40}, "30 previous releases"),
        ({"categories": ["Steam Cloud"]}, "neither single-player nor any multiplayer"),
    ],
)
def test_unusual_inputs_are_flagged(market, overrides, phrase):
    warnings = input_warnings(describe(**overrides), market)
    assert any(phrase in w for w in warnings), warnings
