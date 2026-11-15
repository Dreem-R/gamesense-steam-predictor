import numpy as np
import pandas as pd
import pytest

from gamesense.pipeline.forward_test import evaluate, review_counts_frozen_from


def games(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["release_date", "steam_recommendations", "total_reviews"])
    return frame.assign(release_date=pd.to_datetime(frame["release_date"]))


def test_freeze_is_the_first_month_where_popular_games_have_no_reviews():
    frame = games(
        [
            ("2025-04-10", 500, 480),
            ("2025-04-20", 300, 0),
            ("2025-04-25", 250, 240),
            ("2025-06-05", 400, 0),
            ("2025-06-15", 900, 3),
            ("2025-06-20", 0, 0),
            ("2025-07-01", 350, 1),
        ]
    )
    assert review_counts_frozen_from(frame) == pd.Period("2025-06")


def test_no_freeze_when_counts_keep_up():
    frame = games([("2025-04-10", 500, 480), ("2025-05-10", 300, 310), ("2025-05-20", 0, 0)])
    assert review_counts_frozen_from(frame) is None


def test_evaluate_scores_against_steam_counts():
    frame = pd.DataFrame({"steam_recommendations": [0, 0, 150, 1200]})
    p_success = np.array([0.02, 0.10, 0.50, 0.30])
    p_hit = np.array([0.00, 0.01, 0.10, 0.60])
    result = evaluate(frame, p_success, p_hit)
    assert result["games"] == 4
    assert result["actual_rate"] == pytest.approx(0.5)
    assert result["predicted_rate"] == pytest.approx((0.02 + 0.11 + 0.60 + 0.90) / 4)
    assert result["auc"] == 1
    assert result["hit"]["actual_rate"] == pytest.approx(0.25)
    assert result["hit"]["auc"] == 1
