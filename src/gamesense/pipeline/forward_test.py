"""Check the shipped model on the first release year it has not seen.

The outcome is whether a game reached 100+ reviews, counted by Steam itself. The tiers cannot be checked
this way: they come from SteamSpy's review counts, which stop updating partway through the forward year.

Writes reports/forward_test.json.

Usage: python -m gamesense.pipeline.forward_test
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from gamesense import config
from gamesense.metrics import outcome_metrics
from gamesense.predictor import GameSensePredictor

SUCCESS, HIT = 100, 1000
# A game this popular on Steam with a near-empty SteamSpy count has a stale count, not a bad launch.
POPULAR = 200


def review_counts_frozen_from(games: pd.DataFrame) -> pd.Period | None:
    """First release month in which most popular games have almost no SteamSpy reviews."""
    popular = games[games["steam_recommendations"] >= POPULAR]
    stale = (popular["total_reviews"] < 10).groupby(popular["release_date"].dt.to_period("M")).mean()
    months = stale[stale > 0.5].index
    return months.min() if len(months) else None


def evaluate(games: pd.DataFrame, p_success: np.ndarray, p_hit: np.ndarray) -> dict[str, Any]:
    """Predicted chances against Steam's own review count, for 100+ and for 1,000+ reviews."""
    recommendations = games["steam_recommendations"].to_numpy()
    out = outcome_metrics(recommendations >= SUCCESS, p_success + p_hit)
    hit = outcome_metrics(recommendations >= HIT, p_hit)
    out["hit"] = {k: hit[k] for k in ("actual_rate", "predicted_rate", "auc")}
    return out


def reference_year() -> dict[str, Any]:
    """The held-out test year, scored against the same Steam outcome for comparison."""
    predictions = pd.read_parquet(config.TEST_PREDICTIONS_PATH)
    steam = pd.read_parquet(config.PROCESSED_PATH, columns=["app_id", "steam_recommendations"])
    games = predictions.merge(steam, on="app_id", validate="one_to_one")
    return evaluate(games, games["p_success"].to_numpy(), games["p_hit"].to_numpy())


def main() -> None:
    predictor = GameSensePredictor()
    games = pd.read_parquet(config.FORWARD_PATH)
    proba = predictor.predict_frame(games)
    p_success, p_hit = proba["p_success"].to_numpy(), proba["p_hit"].to_numpy()

    frozen_from = review_counts_frozen_from(games)
    released = games["release_date"].dt.to_period("M")
    before_freeze = (released < frozen_from) if frozen_from is not None else pd.Series(True, index=games.index)
    slices = {
        # The same kind of games the model was trained and tested on.
        "tracked_before_freeze": (games["tracked"] & before_freeze).to_numpy(),
        # What a developer faces: every game with a store page, once it has had time to collect reviews.
        "all_games_mature": (games["months_on_sale"] >= config.MIN_MONTHS_ON_SALE).to_numpy(),
        "all_games": np.ones(len(games), dtype=bool),
    }
    results = {name: evaluate(games[mask], p_success[mask], p_hit[mask]) for name, mask in slices.items()}
    reference = reference_year()
    report = {
        "forward_year": config.FORWARD_YEAR,
        "model_trained_through": predictor.model.prediction_year,
        "outcome": f"{SUCCESS}+ Steam recommendations at the data snapshot",
        "review_counts_frozen_from": str(frozen_from) if frozen_from is not None else None,
        "min_months_on_sale": config.MIN_MONTHS_ON_SALE,
        "reference": {"year": config.TEST_YEARS[0], **reference},
        "slices": results,
    }
    config.FORWARD_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"SteamSpy review counts are stale for releases from {frozen_from}")
    rows = {f"{config.TEST_YEARS[0]} test (reference)": reference, **results}
    table = pd.DataFrame(
        {name: {k: r[k] for k in ("games", "actual_rate", "predicted_rate", "auc")} for name, r in rows.items()}
    ).T
    print(table.round(3).to_string())
    print(f"report -> {config.FORWARD_REPORT_PATH}")


if __name__ == "__main__":
    main()
