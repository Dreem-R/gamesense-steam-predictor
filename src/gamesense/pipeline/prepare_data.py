"""Clean the raw dataset.

Writes data/processed/games_clean.parquet with the releases used for training and evaluation, and
data/processed/games_forward.parquet with the following year's releases for the forward test.

Usage: python -m gamesense.pipeline.prepare_data
"""

import html
import re

import numpy as np
import pandas as pd

from gamesense import config

HTML_TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")


def clean_text(text: str | None, max_chars: int = 3000) -> str:
    text = html.unescape(HTML_TAG.sub(" ", text or ""))
    return SPACES.sub(" ", text).strip()[:max_chars]


def to_list(arr) -> list:
    return [] if arr is None else [str(x) for x in arr]


def prior_history(df: pd.DataFrame, key: str, prefix: str, counted: pd.Series | None = None) -> pd.DataFrame:
    """Number of earlier releases by the same studio and the best review count among them.

    Only rows where `counted` is true add to a studio's history. Every row with a studio name gets one.
    """
    games_col, best_col = f"{prefix}_prior_games", f"{prefix}_prior_best_reviews"
    has_key = (df[key].notna() & (df[key] != "")).to_numpy()
    source = df[has_key] if counted is None else df[has_key & counted.to_numpy()]
    per_day = (
        source.groupby([key, "release_date"])
        .agg(n_games=("total_reviews", "size"), best=("total_reviews", "max"))
        .reset_index()
        .sort_values([key, "release_date"])
    )
    g = per_day.groupby(key)
    per_day[games_col] = g["n_games"].cumsum()
    per_day[best_col] = g["best"].cummax()

    # For each game, the studio's totals on its last release day strictly before this one.
    rows = df.loc[has_key, [key, "release_date"]].assign(_row=np.flatnonzero(has_key))
    matched = pd.merge_asof(
        rows.sort_values("release_date", kind="stable"),
        per_day[[key, "release_date", games_col, best_col]].sort_values("release_date"),
        on="release_date",
        by=key,
        allow_exact_matches=False,
    )
    out = df.reset_index(drop=True)
    for col in (games_col, best_col):
        out[col] = np.nan
        out.loc[matched["_row"].to_numpy(), col] = matched[col].fillna(0).to_numpy()
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Cleaned games from every release year, with studio history and outcome tier.

    Games SteamSpy does not track are kept and marked in `tracked`. Also returns the number of rows each
    filtering step removed.
    """
    df = pd.DataFrame(
        {
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
            # Steam's own count. It is only published once a game passes 100, and is 0 below that.
            "steam_recommendations": raw["recommendations"].fillna(0).astype(int),
            "has_website": (raw["website"].fillna("").str.len() > 0).astype(int),
        }
    )
    df["genres"] = raw["genres"].apply(to_list)
    df["categories"] = raw["categories"].apply(to_list)
    df["developers"] = raw["developers"].apply(to_list)
    df["publishers"] = raw["publishers"].apply(to_list)
    df["n_languages"] = raw["supported_languages"].apply(lambda a: len(to_list(a)))
    df["n_audio_languages"] = raw["full_audio_languages"].apply(lambda a: len(to_list(a)))
    df["n_screenshots"] = raw["screenshots"].apply(lambda a: len(to_list(a)))
    df["short_description"] = raw["short_description"].apply(lambda t: clean_text(t, 400))
    df["about"] = raw["detailed_description"].apply(clean_text)
    df["about_words"] = raw["detailed_description"].apply(lambda t: len(clean_text(t, 10**6).split()))

    # The dataset has no crawl date. Its latest release is within days of it.
    snapshot = df["release_date"].max()
    steps = []

    def keep(mask, why):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        steps.append((why, before - len(df)))

    keep(df["release_date"].notna(), "no release date")
    keep(df["genres"].str.len() > 0, "empty store page (no genres)")
    keep(df["about_words"] > 0, "no description")
    keep(
        ~df["genres"].apply(
            lambda g: (
                bool(set(g) & config.NON_GAME_GENRES)
                and not (
                    set(g) & {"Action", "Adventure", "RPG", "Strategy", "Simulation", "Casual", "Sports", "Racing"}
                )
            )
        ),
        "software, not a game",
    )
    keep(df["price"] <= 100, "price above $100")

    # positive and negative come from SteamSpy, which reports owners '0 - 0' and no reviews for games it
    # does not track.
    df["tracked"] = df["estimated_owners"] != "0 - 0"
    df["total_reviews"] = df["positive"] + df["negative"]

    # History uses every release year, so games from before the training window count. Only tracked games
    # add to it, because the review counts of the others are unknown.
    df["developer"] = df["developers"].apply(lambda a: a[0] if a else "")
    df["publisher"] = df["publishers"].apply(lambda a: a[0] if a else "")
    df = prior_history(df, "developer", "dev", counted=df["tracked"])
    df = prior_history(df, "publisher", "pub", counted=df["tracked"])
    for c in ["dev_prior_games", "dev_prior_best_reviews", "pub_prior_games", "pub_prior_best_reviews"]:
        df[c] = df[c].fillna(0).astype(int)
    df["self_published"] = df.apply(
        lambda r: int(not r["publishers"] or set(r["publishers"]) <= set(r["developers"])), axis=1
    )

    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    df["months_on_sale"] = (snapshot - df["release_date"]).dt.days / 30.44

    df["tier"] = pd.cut(df["total_reviews"], config.TIER_BINS, labels=False).astype(int)
    df["positive_ratio"] = np.where(df["total_reviews"] > 0, df["positive"] / df["total_reviews"].clip(lower=1), np.nan)

    df = df.drop(columns=["developers", "publishers", "estimated_owners"]).reset_index(drop=True)
    return df, steps


def main() -> None:
    raw = pd.read_parquet(config.RAW_PATH)
    print(f"raw rows: {len(raw):,}")
    df, steps = clean(raw)

    in_window = df["year"].between(config.YEAR_MIN, config.YEAR_MAX)
    forward = df["year"] == config.FORWARD_YEAR
    steps.append((f"released outside {config.YEAR_MIN}-{config.FORWARD_YEAR}", int((~in_window & ~forward).sum())))
    steps.append(
        (f"{config.YEAR_MIN}-{config.YEAR_MAX} games not tracked by SteamSpy", int((in_window & ~df["tracked"]).sum()))
    )

    # The tiers come from SteamSpy's review counts, so training uses tracked games only. The forward
    # test is scored against Steam's own count and keeps every game.
    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    window = df[in_window & df["tracked"]].drop(columns="tracked").reset_index(drop=True)
    window.to_parquet(config.PROCESSED_PATH, index=False)
    df[forward].reset_index(drop=True).to_parquet(config.FORWARD_PATH, index=False)

    print("\nfiltering steps:")
    for why, n in steps:
        print(f"  - removed {n:>6,}  {why}")
    print(f"\n{config.YEAR_MIN}-{config.YEAR_MAX} rows: {len(window):,}")
    print(f"{config.FORWARD_YEAR} rows for the forward test: {int(forward.sum()):,}")
    print("tier distribution:")
    print(window["tier"].map(dict(enumerate(config.TIER_NAMES))).value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
