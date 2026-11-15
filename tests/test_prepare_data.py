import pandas as pd

from gamesense.pipeline.prepare_data import clean, clean_text, prior_history


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Fast &amp; <b>fun</b></p>\n\n  game") == "Fast & fun game"


def test_clean_text_truncates_and_handles_none():
    assert clean_text("abcdef", max_chars=3) == "abc"
    assert clean_text(None) == ""


def test_prior_history_only_counts_earlier_releases():
    games = pd.DataFrame(
        {
            "developer": ["A", "A", "A", "A", "B", ""],
            "release_date": pd.to_datetime(
                ["2021-03-01", "2019-01-01", "2019-01-01", "2020-05-01", "2020-01-01", "2020-01-01"]
            ),
            "total_reviews": [3, 10, 50, 5, 900, 7],
        }
    )
    out = prior_history(games, "developer", "dev")

    # Row order is preserved, so results line up with the input.
    assert out["total_reviews"].tolist() == games["total_reviews"].tolist()

    # Two games on the same day do not count each other.
    assert out["dev_prior_games"].tolist()[:5] == [3, 0, 0, 2, 0]
    # Best previous reviews only look at strictly earlier release days.
    assert out["dev_prior_best_reviews"].tolist()[:5] == [50, 0, 0, 50, 0]
    # Games without a developer get no history.
    assert out.iloc[5][["dev_prior_games", "dev_prior_best_reviews"]].isna().all()


def test_prior_history_only_counts_marked_rows():
    games = pd.DataFrame(
        {
            "developer": ["A", "A", "A"],
            "release_date": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"]),
            "total_reviews": [60, 0, 5],
        }
    )
    out = prior_history(games, "developer", "dev", counted=pd.Series([True, False, True]))

    # The uncounted 2020 game still gets a history, but does not add to the 2021 game's.
    assert out["dev_prior_games"].tolist() == [0, 1, 1]
    assert out["dev_prior_best_reviews"].tolist() == [0, 60, 60]


def raw_game(**overrides) -> dict:
    game = {
        "appID": 1,
        "name": "Game",
        "release_date": "2021-01-01",
        "estimated_owners": "0 - 20000",
        "price": 9.99,
        "windows": True,
        "mac": False,
        "linux": False,
        "required_age": 0,
        "achievements": 0,
        "positive": 0,
        "negative": 0,
        "recommendations": 0,
        "website": None,
        "genres": ["Indie"],
        "categories": ["Single-player"],
        "developers": ["Studio"],
        "publishers": ["Studio"],
        "supported_languages": ["English"],
        "full_audio_languages": [],
        "screenshots": ["a.jpg", "b.jpg"],
        "short_description": "A short description.",
        "detailed_description": "<p>A longer description.</p>",
    }
    return game | overrides


def test_clean_keeps_untracked_games_and_marks_them():
    raw = pd.DataFrame(
        [
            raw_game(appID=1, release_date="2019-01-01", positive=50, negative=10),
            raw_game(appID=2, release_date="2020-01-01", estimated_owners="0 - 0"),
            raw_game(appID=3, release_date="2021-01-01", positive=150, recommendations=140),
            raw_game(appID=4, genres=["Utilities"]),
            raw_game(appID=5, detailed_description=""),
            raw_game(appID=6, price=199.0),
        ]
    )
    games, steps = clean(raw)

    assert games["app_id"].tolist() == ["1", "2", "3"]
    assert games["tracked"].tolist() == [True, False, True]
    assert games["steam_recommendations"].tolist() == [0, 0, 140]
    assert games["tier"].tolist() == [1, 0, 2]
    # The untracked 2020 game does not count as a previous release for the 2021 game.
    assert games["dev_prior_games"].tolist() == [0, 1, 1]
    assert games["dev_prior_best_reviews"].tolist() == [0, 60, 60]
    assert games["months_on_sale"].iloc[-1] == 0
    assert dict(steps)["software, not a game"] == 1
    assert dict(steps)["no description"] == 1
    assert dict(steps)["price above $100"] == 1
