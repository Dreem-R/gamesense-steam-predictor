"""Inference: predictions, feature contributions and scored changes."""

# lightgbm must be imported before pyarrow, otherwise prediction segfaults on Windows.
import lightgbm  # noqa: F401, I001

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gamesense import config
from gamesense.bundle import ModelBundle
from gamesense.features import FRIENDLY_NAMES, TEXT_COL, build_base_features, game_text
from gamesense.game import Game, to_frame
from gamesense.suggestions import CombinedPlan, ScoredChange, candidate_changes

MIN_DESCRIPTION_WORDS = 20
YES_NO_FEATURES = {
    "is_free",
    "windows",
    "mac",
    "linux",
    "mature",
    "has_website",
    "self_published",
    "multiplayer_any",
    "controller_any",
}


@dataclass(frozen=True)
class Prediction:
    proba: np.ndarray
    tier: int
    success_chance: float
    reviews_low: float
    reviews_mid: float
    reviews_high: float
    used_text: bool

    @property
    def tier_name(self) -> str:
        return config.TIER_NAMES[self.tier]


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    value: str
    effect: float  # in log reviews

    @property
    def multiplier(self) -> float:
        return float(np.exp(self.effect))


class GameSensePredictor:
    def __init__(self, model: ModelBundle | None = None):
        self.model = model or ModelBundle.load()

    def predict(self, game: Game) -> Prediction:
        _, proba, reviews, used_text = self._score([game])
        lo, mid, hi = sorted([reviews["low"][0], reviews["mid"][0], reviews["high"][0]])
        return Prediction(
            proba=proba[0],
            tier=int(proba[0].argmax()),
            success_chance=float(proba[0, 2:].sum()),
            reviews_low=float(lo),
            reviews_mid=float(mid),
            reviews_high=float(hi),
            used_text=bool(used_text[0]),
        )

    def explain(self, game: Game, top_k: int = 8) -> list[Contribution]:
        """Largest feature contributions from the median review model, with genre flags summed into one."""
        X, _, _, _ = self._score([game])
        contrib = self.model.quantile_models["mid"].predict(X, pred_contrib=True)[0][:-1]
        effects = dict(zip(self.model.feature_cols, contrib, strict=True))
        genre = sum(v for c, v in effects.items() if c.startswith("genre_"))
        # Release year is fixed to the latest training year, so it is not shown.
        effects = {c: v for c, v in effects.items() if not c.startswith("genre_") and c != "year"}
        effects["genres"] = genre
        out = [
            Contribution(
                feature=col,
                label="Genre mix" if col == "genres" else FRIENDLY_NAMES.get(col, col),
                value=describe_value(col, X.iloc[0], game),
                effect=float(val),
            )
            for col, val in effects.items()
        ]
        return sorted(out, key=lambda c: abs(c.effect), reverse=True)[:top_k]

    def what_if(self, game: Game) -> tuple[list[ScoredChange], CombinedPlan | None]:
        """Re-score the game with each candidate change on its own, sorted by the shift in
        P(100+ reviews), plus every helpful low and medium effort change scored together."""
        base = self.predict(game)
        options = candidate_changes(game)
        if not options:
            return [], None

        _, proba, reviews, _ = self._score([c.apply(game) for c in options])
        scored = [
            ScoredChange(
                change=c,
                success_before=base.success_chance,
                success_after=float(proba[i, 2:].sum()),
                reviews_before=base.reviews_mid,
                reviews_after=float(reviews["mid"][i]),
            )
            for i, c in enumerate(options)
        ]
        scored.sort(key=lambda s: (s.uplift, s.reviews_ratio), reverse=True)

        helping = {s.change.id for s in scored if s.helps}
        cheap = [c for c in options if c.effort != "High" and c.id in helping]
        if len(cheap) < 2:
            return scored, None
        combined = game
        for c in cheap:
            combined = c.apply(combined)
        after = self.predict(combined)
        plan = CombinedPlan(
            change_ids=[c.id for c in cheap],
            success_after=after.success_chance,
            reviews_after=after.reviews_mid,
            tier_after=after.tier_name,
        )
        return scored, plan

    def _score(self, games: list[Game]) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], np.ndarray]:
        frame = to_frame(games)
        X, used_text = self._features(frame)
        m = self.model
        proba = m.w_xgb * np.asarray(m.xgb.predict_proba(X)) + (1 - m.w_xgb) * np.asarray(m.lgb.predict_proba(X))
        reviews = {k: np.expm1(q.predict(X)).clip(min=0) for k, q in m.quantile_models.items()}
        return X, proba, reviews, used_text

    def _features(self, games: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = build_base_features(games)
        words = (games["short_description"].fillna("") + " " + games["about"].fillna("")).str.count(r"\S+")
        used_text = (words >= MIN_DESCRIPTION_WORDS).to_numpy()
        score = self.model.text_scorer.predict(game_text(games))
        X[TEXT_COL] = np.where(used_text, score, self.model.feature_medians[TEXT_COL])
        return X[self.model.feature_cols], used_text


def describe_value(col: str, x: pd.Series, game: Game) -> str:
    if col == "genres":
        return ", ".join(game.genres)
    if col == TEXT_COL:
        return "from your description"
    v = x[col]
    if col == "about_words":
        return f"{game.about_words} words"
    if col in {"dev_prior_best_reviews", "pub_prior_best_reviews", "pub_prior_games"}:
        return f"{int(round(np.expm1(v))):,}"
    if col == "price":
        return "Free" if v == 0 else f"${v:.2f}"
    if col.startswith("cat_") or col in YES_NO_FEATURES:
        return "Yes" if v else "No"
    return f"{v:g}"
