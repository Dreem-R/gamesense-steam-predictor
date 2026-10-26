import pandas as pd

from gamesense.pipeline.prepare_data import clean_text, prior_history


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
