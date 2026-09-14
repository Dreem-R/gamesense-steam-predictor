"""Feature engineering shared by training and the Streamlit app.

Only information available before launch is used.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src import config


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


GENRE_COLS = [f"genre_{slug(g)}" for g in config.GENRES]
CATEGORY_COLS = [f"cat_{slug(c)}" for c in config.CATEGORIES]

BASE_COLS = [
    "price", "is_free", "windows", "mac", "linux", "mature",
    "achievements", "n_languages", "n_audio_languages", "n_screenshots",
    "has_website", "about_words", "short_desc_chars", "year", "month",
    "dev_prior_games", "dev_prior_best_reviews", "pub_prior_games",
    "pub_prior_best_reviews", "self_published", "n_genres", "n_categories",
    "multiplayer_any", "controller_any",
]
TEXT_COL = "text_score"
FEATURE_COLS = BASE_COLS + GENRE_COLS + CATEGORY_COLS + [TEXT_COL]

FRIENDLY_NAMES = {
    "price": "Price", "is_free": "Free to play", "windows": "Windows", "mac": "macOS",
    "linux": "Linux / Steam Deck", "mature": "Mature (17+)", "achievements": "Achievements",
    "n_languages": "Languages supported", "n_audio_languages": "Fully voiced languages",
    "n_screenshots": "Screenshots on store page", "has_website": "Has a website",
    "about_words": "Store description length", "short_desc_chars": "Short description length",
    "year": "Release year", "month": "Release month",
    "dev_prior_games": "Studio's previous games", "dev_prior_best_reviews": "Studio's best previous game (reviews)",
    "pub_prior_games": "Publisher's catalogue size", "pub_prior_best_reviews": "Publisher's best game (reviews)",
    "self_published": "Self-published", "n_genres": "Number of genres",
    "n_categories": "Number of Steam features", "multiplayer_any": "Any multiplayer",
    "controller_any": "Controller support", TEXT_COL: "Description wording (NLP score)",
}
FRIENDLY_NAMES.update({f"genre_{slug(g)}": f"Genre: {g}" for g in config.GENRES})
FRIENDLY_NAMES.update({f"cat_{slug(c)}": f"Feature: {c}" for c in config.CATEGORIES})


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric feature matrix from rows in the clean-table schema."""
    X = pd.DataFrame(index=df.index)
    X["price"] = df["price"].astype(float)
    X["is_free"] = (df["price"] == 0).astype(int)
    for c in ["windows", "mac", "linux", "has_website", "self_published", "year", "month"]:
        X[c] = df[c].astype(int)
    X["mature"] = (df["required_age"] >= 17).astype(int)
    X["achievements"] = df["achievements"].clip(0, 1000)
    X["n_languages"] = df["n_languages"].clip(0, 40)
    X["n_audio_languages"] = df["n_audio_languages"].clip(0, 30)
    X["n_screenshots"] = df["n_screenshots"].clip(0, 40)
    X["about_words"] = np.log1p(df["about_words"])
    X["short_desc_chars"] = df["short_description"].str.len()
    X["dev_prior_games"] = df["dev_prior_games"].clip(0, 100)
    X["dev_prior_best_reviews"] = np.log1p(df["dev_prior_best_reviews"])
    X["pub_prior_games"] = np.log1p(df["pub_prior_games"])
    X["pub_prior_best_reviews"] = np.log1p(df["pub_prior_best_reviews"])

    genres = df["genres"].apply(set)
    cats = df["categories"].apply(set)
    for g, col in zip(config.GENRES, GENRE_COLS):
        X[col] = genres.apply(lambda s: int(g in s))
    for c, col in zip(config.CATEGORIES, CATEGORY_COLS):
        X[col] = cats.apply(lambda s: int(c in s))
    X["n_genres"] = X[GENRE_COLS].sum(axis=1)
    X["n_categories"] = X[CATEGORY_COLS].sum(axis=1)
    mp = [slug(c) for c in ["Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op", "MMO"]]
    X["multiplayer_any"] = X[[f"cat_{m}" for m in mp]].max(axis=1)
    X["controller_any"] = X[["cat_full_controller_support", "cat_partial_controller_support"]].max(axis=1)
    return X[BASE_COLS + GENRE_COLS + CATEGORY_COLS]


# Wording typically added to a store page after launch (awards, sales figures,
# editions, patch notes). Stripped to avoid target leakage.
POST_LAUNCH_HYPE = re.compile(
    r"\b(acclaimed|award\w*|winn\w*|nominat\w*|goty|game of the year|best[- ]?sell\w*|"
    r"million\w*|sold|copies|overwhelmingly|positive reviews|reviews?|rated|metacritic|"
    r"editions?|deluxe|definitive|remaster\w*|complete|bundle|soundtrack|dlcs?|season pass|"
    r"expansions?|updates?|updated|patch\w*|version|sequel|franchise|iconic|legendary|"
    r"\d+(\.\d+)?\s*[km]\+?|\d{1,3}(,\d{3})+)\b"
)


def game_text(df: pd.DataFrame) -> pd.Series:
    text = (df["short_description"].fillna("") + " " + df["about"].fillna("")).str.lower()
    return text.str.replace(POST_LAUNCH_HYPE, " ", regex=True)


class TextScorer:
    """TF-IDF + Ridge on the store description, predicting log1p(reviews).
    The prediction is used as a stacked feature."""

    def __init__(self, max_features: int = 40000, alpha: float = 3.0):
        self.max_features = max_features
        self.alpha = alpha

    def _new(self):
        vec = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), min_df=5,
                              max_df=0.6, sublinear_tf=True, stop_words="english", dtype=np.float32)
        return vec, Ridge(alpha=self.alpha)

    def fit(self, texts: pd.Series, y: np.ndarray):
        self.vec, self.model = self._new()
        self.model.fit(self.vec.fit_transform(texts), y)
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        return self.model.predict(self.vec.transform(texts))

    def fit_oof(self, texts: pd.Series, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
        """Return out-of-fold predictions for the given rows, then refit on all of them."""
        oof = np.zeros(len(texts))
        for tr, va in KFold(n_splits, shuffle=True, random_state=config.RANDOM_STATE).split(texts):
            vec, model = self._new()
            model.fit(vec.fit_transform(texts.iloc[tr]), y[tr])
            oof[va] = model.predict(vec.transform(texts.iloc[va]))
        self.fit(texts, y)
        return oof

    def top_terms(self, k: int = 25):
        terms = np.array(self.vec.get_feature_names_out())
        coef = self.model.coef_
        order = np.argsort(coef)
        return list(zip(terms[order[-k:][::-1]], coef[order[-k:][::-1]])), list(zip(terms[order[:k]], coef[order[:k]]))
