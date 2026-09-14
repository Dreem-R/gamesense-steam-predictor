"""Inference wrapper: predictions, feature contributions and suggested changes."""
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

    def what_if(self, game: dict) -> tuple[pd.DataFrame, dict]:
        """Re-score the game with single changes.

        Returns one row per change, sorted by the shift in P(100+ reviews), and a
        combined plan that applies every low and medium effort change that helped.
        A change helps if it adds at least 1 point of success chance or 15% more
        expected reviews, so games with very low or very high chances still get
        useful comparisons.
        """
        base = self.predict(game)
        options = candidate_changes(game)
        if not options:
            return pd.DataFrame(columns=SUGGESTION_COLS), {}

        frame = pd.concat([to_frame(dict(game, **o["changes"])) for o in options], ignore_index=True)
        _, proba, _, q, _ = self.predict_frame(frame)
        out = pd.DataFrame(options).drop(columns="changes")
        out["success_before"] = base.success_chance
        out["success_after"] = proba[:, 2:].sum(1)
        out["uplift"] = out.success_after - base.success_chance
        out["reviews_before"] = base.reviews_mid
        out["reviews_after"] = q["mid"]
        out["reviews_ratio"] = (q["mid"] + 1) / (base.reviews_mid + 1)
        out["helps"] = (out.uplift >= MIN_UPLIFT) | ((out.reviews_ratio >= MIN_REVIEW_RATIO) & (out.uplift > -0.005))
        out = out.sort_values(["uplift", "reviews_ratio"], ascending=False).reset_index(drop=True)

        helping = set(out.loc[out.helps, "change"])
        helpful = [o for o in options if o["effort"] != "High" and o["change"] in helping]
        plan = {}
        if len(helpful) > 1:
            combined = dict(game)
            for o in helpful:
                changes = dict(o["changes"])
                if "categories" in changes:
                    changes["categories"] = sorted(set(combined["categories"]) | set(changes["categories"]))
                combined.update(changes)
            after = self.predict(combined)
            plan = {"changes": [o["change"] for o in helpful], "success_after": after.success_chance,
                    "reviews_after": after.reviews_mid, "tier_after": self.tier_names[after.tier]}
        return out, plan


MIN_UPLIFT = 0.01
MIN_REVIEW_RATIO = 1.15
SUGGESTION_COLS = ["change", "detail", "area", "effort", "success_before", "success_after", "uplift",
                   "reviews_before", "reviews_after", "reviews_ratio", "helps"]


def candidate_changes(game: dict) -> list[dict]:
    cats = set(game["categories"])
    options = []

    def add(change, detail, area, effort, **changes):
        options.append({"change": change, "detail": detail, "area": area, "effort": effort, "changes": changes})

    if game["n_screenshots"] < 10:
        add("Show at least 10 screenshots",
            "Cover different areas, mechanics and moods. Steam uses them in search previews and on the store page.",
            "Store page", "Low", n_screenshots=10)
    if not game["has_website"]:
        add("Put up a game website",
            "A single page with the trailer, a press kit and store links is enough.",
            "Store page", "Low", has_website=1)
    if "Steam Cloud" not in cats:
        add("Enable Steam Cloud saves",
            "Mostly Steamworks configuration. Lets players continue on another PC or a Steam Deck.",
            "Steam features", "Low", categories=sorted(cats | {"Steam Cloud"}))
    if game["achievements"] == 0:
        add("Add around 25 Steam Achievements",
            "Gives players goals and puts the game in friends' activity feeds.",
            "Steam features", "Medium", achievements=25, categories=sorted(cats | {"Steam Achievements"}))
    if not cats & {"Full controller support", "Partial Controller Support"}:
        add("Add full controller support",
            "Needed for a Steam Deck Verified rating and for couch players.",
            "Steam features", "Medium", categories=sorted(cats | {"Full controller support"}))
    if game["n_languages"] < 8:
        add("Localise into 8 languages",
            "Common first targets are Simplified Chinese, Russian, Spanish, Brazilian Portuguese, German, "
            "French and Japanese. Costs depend heavily on how much text the game has.",
            "Reach", "Medium", n_languages=8)
    if not game["linux"]:
        add("Ship a native Linux build",
            "Helps with Linux players and Steam Deck. Engines like Unity and Godot can export Linux builds.",
            "Reach", "Medium", linux=1)
    if not game["mac"]:
        add("Ship a macOS build",
            "Requires testing on Apple hardware and notarisation.",
            "Reach", "Medium", mac=1)
    if 0 < game["price"] < 10:
        add(f"Price at $14.99 instead of ${game['price']:.2f}",
            "Very cheap games are often read as low effort. Only worth it if the content supports the price.",
            "Pricing", "Low", price=14.99)
    if game["price"] > 30:
        add(f"Price at $24.99 instead of ${game['price']:.2f}",
            "Few games from smaller studios sell well above $30.",
            "Pricing", "Low", price=24.99)
    if not cats & {"Co-op", "Online Co-op"}:
        add("Add online co-op",
            "A design change with networking work. Only consider it if it fits the game.",
            "Design", "High", categories=sorted(cats | {"Multi-player", "Co-op", "Online Co-op"}))
    if game["self_published"]:
        add("Work with an established publisher",
            "Tested as a publisher with 50+ released games. Publishers take a revenue share in return "
            "for marketing, funding or porting.",
            "Business", "High", self_published=0, pub_prior_games=50, pub_prior_best_reviews=5000)
    return options


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
