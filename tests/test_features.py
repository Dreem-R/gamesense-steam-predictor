import numpy as np
import pandas as pd
import pytest

from gamesense import config
from gamesense.features import (
    BASE_COLS,
    CATEGORY_COLS,
    FEATURE_COLS,
    GENRE_COLS,
    TEXT_COL,
    build_base_features,
    game_text,
    slug,
)
from gamesense.predictor import default_game, to_frame


def features_for(**overrides) -> pd.Series:
    return build_base_features(to_frame(default_game(**overrides))).iloc[0]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Free To Play", "free_to_play"), ("Shared/Split Screen", "shared_split_screen"), ("  Co-op ", "co_op")],
)
def test_slug(name, expected):
    assert slug(name) == expected


def test_feature_columns_are_unique_and_ordered():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert FEATURE_COLS == BASE_COLS + GENRE_COLS + CATEGORY_COLS + [TEXT_COL]
    assert len(GENRE_COLS) == len(config.GENRES)
    assert len(CATEGORY_COLS) == len(config.CATEGORIES)


def test_base_features_have_every_column_except_text():
    X = build_base_features(to_frame(default_game()))
    assert list(X.columns) == BASE_COLS + GENRE_COLS + CATEGORY_COLS
    assert not X.isna().any().any()


def test_price_and_age_flags():
    assert features_for(price=0.0)["is_free"] == 1
    assert features_for(price=4.99)["is_free"] == 0
    assert features_for(required_age=17)["mature"] == 1
    assert features_for(required_age=16)["mature"] == 0


def test_counts_are_clipped():
    x = features_for(n_languages=120, n_screenshots=500, achievements=5000, dev_prior_games=999)
    assert x["n_languages"] == 40
    assert x["n_screenshots"] == 40
    assert x["achievements"] == 1000
    assert x["dev_prior_games"] == 100


def test_skewed_counts_are_log_scaled():
    x = features_for(about_words=999, pub_prior_games=99, dev_prior_best_reviews=9999)
    assert x["about_words"] == pytest.approx(np.log1p(999))
    assert x["pub_prior_games"] == pytest.approx(np.log1p(99))
    assert x["dev_prior_best_reviews"] == pytest.approx(np.log1p(9999))


def test_genre_and_category_flags():
    cats = ["Single-player", "Online Co-op", "Partial Controller Support"]
    x = features_for(genres=["Indie", "RPG"], categories=cats)
    assert x["genre_indie"] == 1
    assert x["genre_rpg"] == 1
    assert x["genre_action"] == 0
    assert x["n_genres"] == 2
    assert x["cat_online_co_op"] == 1
    assert x["n_categories"] == 3
    assert x["multiplayer_any"] == 1
    assert x["controller_any"] == 1


def test_unknown_labels_are_ignored():
    x = features_for(genres=["Indie", "Not A Genre"], categories=["Single-player", "Steam Trading Cards"])
    assert x["n_genres"] == 1
    assert x["n_categories"] == 1


def test_singleplayer_only_game_has_no_multiplayer_or_controller():
    x = features_for(categories=["Single-player"])
    assert x["multiplayer_any"] == 0
    assert x["controller_any"] == 0


def test_game_text_removes_post_launch_wording():
    df = pd.DataFrame(
        {
            "short_description": ["Award-winning roguelike, now with 2M players!"],
            "about": ["Sold 1,500,000 copies. Get the Deluxe Edition and the soundtrack in the latest update."],
        }
    )
    text = game_text(df).iloc[0]
    for leaked in ["award", "2m", "sold", "1,500,000", "copies", "deluxe", "edition", "soundtrack", "update"]:
        assert leaked not in text.split(), leaked
    for kept in ["roguelike", "players", "latest"]:
        assert kept in text


def test_game_text_handles_missing_values():
    df = pd.DataFrame({"short_description": [None], "about": ["A puzzle game"]})
    assert game_text(df).iloc[0].strip() == "a puzzle game"
