"""Inference wrapper: predictions, feature contributions and what-if scenarios."""
# lightgbm must be imported before pyarrow, otherwise prediction segfaults on Windows.
import lightgbm  # noqa: F401, I001

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import FRIENDLY_NAMES, TEXT_COL, build_base_features, game_text

MIN_DESCRIPTION_WORDS = 20


@dataclass
class Prediction:
    proba: np.ndarray
    tier: int
    success_chance: float
    reviews_low: float
    reviews_mid: float
    reviews_high: float
    used_text: bool


class GameSensePredictor:
    def __init__(self, path=config.MODELS_DIR / "gamesense.joblib"):
        b = joblib.load(path)
        self.b = b
        self.cols = b["feature_cols"]
        self.tier_names = b["tier_names"]

    def features(self, games: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = build_base_features(games)
        words = (games["short_description"].fillna("") + " " + games["about"].fillna("")).str.split().str.len()
        used_text = (words >= MIN_DESCRIPTION_WORDS).values
        score = self.b["text_scorer"].predict(game_text(games))
        X[TEXT_COL] = np.where(used_text, score, self.b["feature_medians"][TEXT_COL])
        return X[self.cols], used_text

    def predict_frame(self, games: pd.DataFrame):
        X, used_text = self.features(games)
        w = self.b["w_xgb"]
        proba = w * self.b["xgb"].predict_proba(X) + (1 - w) * self.b["lgb"].predict_proba(X)
        q = {k: np.expm1(m.predict(X)).clip(min=0) for k, m in self.b["quantile_models"].items()}
        tiers = proba.argmax(1)
        return X, proba, tiers, q, used_text

    def predict(self, game: dict) -> Prediction:
        _, proba, tiers, q, used_text = self.predict_frame(to_frame(game))
        lo, mid, hi = sorted([q["low"][0], q["mid"][0], q["high"][0]])
        return Prediction(proba[0], int(tiers[0]), float(proba[0, 2:].sum()), lo, mid, hi, bool(used_text[0]))

    def explain(self, game: dict, top_k: int = 8) -> pd.DataFrame:
        """Feature contributions from the median quantile model, in log-reviews.
        Genre flags are summed into one row."""
        X, _ = self.features(to_frame(game))
        contrib = self.b["quantile_models"]["mid"].predict(X, pred_contrib=True)[0][:-1]
        s = pd.Series(contrib, index=self.cols)
        genre = s[[c for c in self.cols if c.startswith("genre_")]].sum()
        # Release year is fixed to the latest training year, so it is not shown.
        s = s[[c for c in self.cols if not c.startswith("genre_") and c != "year"]]
        s["genres"] = genre
        rows = []
        for col, val in s.items():
            label = "Genre mix" if col == "genres" else FRIENDLY_NAMES.get(col, col)
            rows.append({"feature": col, "label": label, "value": describe_value(col, X.iloc[0], game),
                         "effect": float(val), "multiplier": float(np.exp(val))})
        out = pd.DataFrame(rows)
        out = out.reindex(out.effect.abs().sort_values(ascending=False).index).head(top_k)
        return out.reset_index(drop=True)

    def what_if(self, game: dict) -> pd.DataFrame:
        """Re-score the game with single changes and report the shift in P(100+ reviews)."""
        base = self.predict(game).success_chance
        variants = []

        def add(label, **changes):
            g = dict(game, **changes)
            if g != game:
                variants.append((label, g))

        cats = set(game["categories"])
        if game["n_languages"] < 8:
            add("Localise into 8 languages", n_languages=8)
        if game["n_screenshots"] < 12:
            add("Show 12 screenshots on the store page", n_screenshots=12)
        if not game["linux"]:
            add("Support Linux / Steam Deck (Proton-native)", linux=1)
        if not game["mac"]:
            add("Release on macOS", mac=1)
        if game["achievements"] == 0:
            add("Add ~25 Steam Achievements", achievements=25,
                categories=sorted(cats | {"Steam Achievements"}))
        if not cats & {"Full controller support", "Partial Controller Support"}:
            add("Add full controller support", categories=sorted(cats | {"Full controller support"}))
        if "Steam Cloud" not in cats:
            add("Enable Steam Cloud saves", categories=sorted(cats | {"Steam Cloud"}))
        if not cats & {"Co-op", "Online Co-op"}:
            add("Add online co-op", categories=sorted(cats | {"Multi-player", "Co-op", "Online Co-op"}))
        if not game["has_website"]:
            add("Create a game website", has_website=1)
        if game["about_words"] < 250:
            add("Write a fuller store description (~300 words)", about_words=300)
        if 0 < game["price"] < 10:
            add(f"Price at $14.99 instead of ${game['price']:.2f}", price=14.99)
        if game["price"] > 30:
            add(f"Price at $24.99 instead of ${game['price']:.2f}", price=24.99)
        if game["self_published"]:
            add("Sign with an established publisher (50+ games)", self_published=0,
                pub_prior_games=50, pub_prior_best_reviews=5000)

        if not variants:
            return pd.DataFrame(columns=["change", "success_chance", "uplift"])
        frame = pd.concat([to_frame(g) for _, g in variants], ignore_index=True)
        _, proba, _, _, _ = self.predict_frame(frame)
        out = pd.DataFrame({"change": [v[0] for v in variants], "success_chance": proba[:, 2:].sum(1)})
        out["uplift"] = out.success_chance - base
        return out.sort_values("uplift", ascending=False).reset_index(drop=True)


def to_frame(game: dict) -> pd.DataFrame:
    return pd.DataFrame([game])


def describe_value(col: str, x: pd.Series, game: dict) -> str:
    if col == "genres":
        return ", ".join(game["genres"]) or "none"
    if col == TEXT_COL:
        return "from your description"
    v = x[col]
    if col in {"about_words"}:
        return f"{game['about_words']} words"
    if col in {"dev_prior_best_reviews", "pub_prior_best_reviews", "pub_prior_games"}:
        return f"{int(round(np.expm1(v))):,}"
    if col == "price":
        return "Free" if v == 0 else f"${v:.2f}"
    if col.startswith("cat_") or col in {"is_free", "windows", "mac", "linux", "mature", "has_website",
                                          "self_published", "multiplayer_any", "controller_any"}:
        return "Yes" if v else "No"
    return f"{v:g}"


def default_game(**overrides) -> dict:
    g = dict(
        name="My Game", short_description="", about="", about_words=0, price=9.99,
        genres=["Indie"], categories=["Single-player"], windows=1, mac=0, linux=0,
        required_age=0, achievements=0, n_languages=1, n_audio_languages=0, n_screenshots=6,
        has_website=0, year=config.YEAR_MAX, month=10, dev_prior_games=0, dev_prior_best_reviews=0,
        pub_prior_games=0, pub_prior_best_reviews=0, self_published=1,
    )
    g.update(overrides)
    if "about_words" not in overrides:
        g["about_words"] = len(g["about"].split())
    return g
