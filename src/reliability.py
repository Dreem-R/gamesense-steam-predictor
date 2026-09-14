"""Reliability checks for a single prediction, based on held-out 2024 results."""
import numpy as np
import pandas as pd

from src import config


def confidence_level(proba: np.ndarray) -> tuple[str, str]:
    order = np.argsort(proba)[::-1]
    top, second = proba[order[0]], proba[order[1]]
    names = config.TIER_NAMES
    if top >= 0.6:
        return "High", f"{top:.0%} probability on {names[order[0]]}."
    if top - second >= 0.2:
        return "Moderate", f"{top:.0%} on {names[order[0]]}, next is {names[order[1]]} at {second:.0%}."
    return "Low", f"Split between {names[order[0]]} ({top:.0%}) and {names[order[1]]} ({second:.0%})."


def tier_track_record(test: pd.DataFrame, tier: int) -> dict:
    """What actually happened to 2024 games that the model placed in the same tier."""
    same = test[test.pred_tier == tier]
    return {
        "games": len(same),
        "exact": float((same.tier == tier).mean()),
        "within_one": float((np.abs(same.tier - tier) <= 1).mean()),
        "actual_mix": same.tier.value_counts(normalize=True).reindex(range(4), fill_value=0).tolist(),
    }


def calibration_check(test: pd.DataFrame, success_chance: float, width: float = 0.05) -> dict:
    """Share of 2024 games with a similar predicted chance that really reached 100+ reviews."""
    p = test.p_success + test.p_hit
    lo, hi = max(0.0, success_chance - width), min(1.0, success_chance + width)
    near = test[(p >= lo) & (p <= hi)]
    return {"games": len(near), "low": lo, "high": hi,
            "actual_rate": float((near.tier >= 2).mean()) if len(near) else float("nan")}


def input_warnings(game: dict, market: pd.DataFrame) -> list[str]:
    """Flags inputs that are rare in the training data, where predictions are less reliable."""
    notes = []
    if game["genres"]:
        similar = market.genres.apply(lambda g: set(game["genres"]) <= set(g)).sum()
        if similar < 100:
            notes.append(f"Only {similar} games in the data share this exact genre combination, "
                         "so the model has few comparable examples.")
    if game["about_words"] < 20:
        notes.append("There is no real store description, so the text part of the model is not used.")
    if game["price"] > 60:
        notes.append("Very few games in the data are priced above $60.")
    if game["n_languages"] > 20:
        notes.append("More than 20 languages is rare outside large publishers.")
    if game["achievements"] > 300:
        notes.append("More than 300 achievements is unusual and often linked to achievement-farming games.")
    if game["dev_prior_games"] > 30:
        notes.append("Studios with more than 30 previous releases are rare and behave differently.")
    multiplayer = {"Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op", "MMO"}
    if "Single-player" not in game["categories"] and not set(game["categories"]) & multiplayer:
        notes.append("The game has neither single-player nor any multiplayer mode selected.")
    return notes
