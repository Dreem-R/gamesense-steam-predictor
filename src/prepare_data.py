"""Clean the raw dataset and write data/processed/games_clean.parquet.

Usage: python -m src.prepare_data
"""
import html
import re

import numpy as np
import pandas as pd

from src import config

HTML_TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")


def clean_text(text: str, max_chars: int = 3000) -> str:
    text = html.unescape(HTML_TAG.sub(" ", text or ""))
    return SPACES.sub(" ", text).strip()[:max_chars]


def to_list(arr) -> list:
    return [] if arr is None else [str(x) for x in arr]


def prior_history(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Number of earlier releases by the same studio and the best review count among them."""
    d = df[[key, "release_date", "total_reviews"]].dropna(subset=[key])
    d = d[d[key] != ""]
    per_day = (
        d.groupby([key, "release_date"])
        .agg(n_games=("total_reviews", "size"), best=("total_reviews", "max"))
        .reset_index()
        .sort_values([key, "release_date"])
    )
    g = per_day.groupby(key)
    per_day[f"{prefix}_prior_games"] = g["n_games"].cumsum() - per_day["n_games"]
    per_day[f"{prefix}_prior_best_reviews"] = (
        g["best"].cummax().groupby(per_day[key]).shift(1).fillna(0)
    )
    cols = [key, "release_date", f"{prefix}_prior_games", f"{prefix}_prior_best_reviews"]
    return df.merge(per_day[cols], on=[key, "release_date"], how="left")


def main() -> None:
    raw = pd.read_parquet(config.RAW_PATH)
    print(f"raw rows: {len(raw):,}")

    df = pd.DataFrame({
        "app_id": raw["appID"].astype(str),
        "name": raw["name"].fillna(""),
        "release_date": pd.to_datetime(raw["release_date"], format="mixed", errors="coerce"),
        "estimated_owners": raw["estimated_owners"],
        "price": raw["price"].astype(float),
        "windows": raw["windows"].astype(int),
        "mac": raw["mac"].astype(int),
        "linux": raw["linux"].astype(int),
        "required_age": raw["required_age"].astype(int),
        "achievements": raw["achievements"].astype(int),
        "positive": raw["positive"].astype(int),
        "negative": raw["negative"].astype(int),
        "has_website": (raw["website"].fillna("").str.len() > 0).astype(int),
    })
    df["genres"] = raw["genres"].apply(to_list)
    df["categories"] = raw["categories"].apply(to_list)
    df["developers"] = raw["developers"].apply(to_list)
    df["publishers"] = raw["publishers"].apply(to_list)
    df["n_languages"] = raw["supported_languages"].apply(lambda a: len(to_list(a)))
    df["n_audio_languages"] = raw["full_audio_languages"].apply(lambda a: len(to_list(a)))
    df["n_screenshots"] = raw["screenshots"].apply(lambda a: len(to_list(a)))
    df["short_description"] = raw["short_description"].apply(lambda t: clean_text(t, 400))
    df["about"] = raw["detailed_description"].apply(clean_text)
    df["about_words"] = raw["detailed_description"].apply(
        lambda t: len(clean_text(t, 10**6).split())
    )

    steps = []

    def keep(mask, why):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        steps.append((why, before - len(df)))

    keep(df["release_date"].notna(), "no release date")
    keep(df["genres"].str.len() > 0, "empty store page (no genres)")
    keep(df["about_words"] > 0, "no description")
    keep(df["estimated_owners"] != "0 - 0", "not tracked by SteamSpy (owners '0 - 0')")
    keep(~df["genres"].apply(lambda g: bool(set(g) & config.NON_GAME_GENRES) and not
                             (set(g) & {"Action", "Adventure", "RPG", "Strategy",
                                        "Simulation", "Casual", "Sports", "Racing"})),
         "software, not a game")
    keep(df["price"] <= 100, "price above $100")

    df["total_reviews"] = df["positive"] + df["negative"]

    # History is computed before the year filter so pre-2018 releases count.
    df["developer"] = df["developers"].apply(lambda a: a[0] if a else "")
    df["publisher"] = df["publishers"].apply(lambda a: a[0] if a else "")
    df = prior_history(df, "developer", "dev")
    df = prior_history(df, "publisher", "pub")
    for c in ["dev_prior_games", "dev_prior_best_reviews", "pub_prior_games", "pub_prior_best_reviews"]:
        df[c] = df[c].fillna(0).astype(int)
    df["self_published"] = df.apply(
        lambda r: int(not r["publishers"] or set(r["publishers"]) <= set(r["developers"])), axis=1
    )

    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    keep(df["year"].between(config.YEAR_MIN, config.YEAR_MAX),
         f"released outside {config.YEAR_MIN}-{config.YEAR_MAX}")

    df["tier"] = pd.cut(df["total_reviews"], config.TIER_BINS, labels=False).astype(int)
    df["positive_ratio"] = np.where(df["total_reviews"] > 0, df["positive"] / df["total_reviews"].clip(lower=1), np.nan)

    df = df.drop(columns=["developers", "publishers", "estimated_owners"]).reset_index(drop=True)
    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PATH, index=False)

    print("\nfiltering steps:")
    for why, n in steps:
        print(f"  - removed {n:>6,}  {why}")
    print(f"\nclean rows: {len(df):,}")
    print("tier distribution:")
    print(df["tier"].map(dict(enumerate(config.TIER_NAMES))).value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
