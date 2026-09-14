"""Exploratory figures and the reduced dataset used by the app's insights page.

Usage: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

APP_DATA = config.ROOT / "app" / "data" / "insights.parquet"
PALETTE = ["#d1495b", "#edae49", "#66a182", "#2e86ab"]


def success_rate(df, by):
    return (df.groupby(by, observed=True)
              .agg(games=("tier", "size"), success_rate=("tier", lambda t: (t >= 2).mean()),
                   median_reviews=("total_reviews", "median"))
              .reset_index())


def light_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["name", "year", "month", "price", "total_reviews", "positive_ratio", "tier",
            "n_languages", "n_screenshots", "achievements", "dev_prior_games", "self_published",
            "linux", "mac", "genres", "categories"]
    out = df[keep].copy()
    out["genres"] = out["genres"].apply(lambda a: "|".join(a))
    out["categories"] = out["categories"].apply(lambda a: "|".join(a))
    return out


def main():
    df = pd.read_parquet(config.PROCESSED_PATH)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # 1. Tier distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df.tier.value_counts(normalize=True).sort_index()
    ax.bar(config.TIER_NAMES, counts.values * 100, color=PALETTE)
    for i, v in enumerate(counts.values):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center")
    ax.set_ylabel("% of games")
    ax.set_title(f"How {len(df):,} Steam games (2018-2024) actually did")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "01_tier_distribution.png", dpi=150); plt.close(fig)

    # 2. Market saturation by year
    by_year = df.groupby("year").agg(games=("tier", "size"), success=("tier", lambda t: (t >= 2).mean()))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(by_year.index, by_year.games, color="#c9d6df")
    ax1.set_ylabel("games released")
    ax2 = ax1.twinx()
    ax2.plot(by_year.index, by_year.success * 100, color="#d1495b", marker="o", lw=2.5)
    ax2.set_ylabel("% reaching 100+ reviews", color="#d1495b"); ax2.grid(False)
    ax1.set_title("More games every year, fewer of them break through")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "02_saturation_by_year.png", dpi=150); plt.close(fig)

    # 3. Success rate by genre
    ex = df[["genres", "tier", "total_reviews"]].explode("genres")
    g = success_rate(ex, "genres")
    g = g[g.games >= 400].sort_values("success_rate")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(g.genres, g.success_rate * 100, color="#2e86ab")
    ax.axvline((df.tier >= 2).mean() * 100, color="k", ls="--", lw=1, label="all games")
    ax.set_xlabel("% of games reaching 100+ reviews"); ax.legend()
    ax.set_title("Success rate by genre")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "03_success_by_genre.png", dpi=150); plt.close(fig)

    # 4. Levers a developer controls
    levers = {
        "Price": pd.cut(df.price, [-0.01, 0, 4.99, 9.99, 19.99, 29.99, 100],
                        labels=["Free", "<$5", "$5-10", "$10-20", "$20-30", "$30+"]),
        "Languages supported": pd.cut(df.n_languages, [-1, 1, 3, 7, 12, 100], labels=["1", "2-3", "4-7", "8-12", "13+"]),
        "Screenshots": pd.cut(df.n_screenshots, [-1, 4, 7, 10, 15, 200], labels=["0-4", "5-7", "8-10", "11-15", "16+"]),
        "Studio's previous games": pd.cut(df.dev_prior_games, [-1, 0, 1, 3, 10, 1000], labels=["0", "1", "2-3", "4-10", "11+"]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (title, bins) in zip(axes.flat, levers.items()):
        s = success_rate(df.assign(bin=bins), "bin")
        ax.bar(s["bin"].astype(str), s.success_rate * 100, color="#66a182")
        ax.set_title(title); ax.set_ylabel("% reaching 100+ reviews")
    fig.suptitle("What developers control vs. how often games succeed", fontsize=14)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "04_levers.png", dpi=150); plt.close(fig)

    # 5. Review count distribution (log scale)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(df.total_reviews + 1), bins=60, color="#2e86ab")
    for b in [10, 100, 1000]:
        ax.axvline(np.log10(b + 1), color="k", ls="--", lw=1)
    ax.set_xticks([0, 1, 2, 3, 4, 5], ["1", "10", "100", "1k", "10k", "100k"])
    ax.set_xlabel("total reviews (log scale) - dashed lines = tier boundaries")
    ax.set_ylabel("games"); ax.set_title("Steam is a long tail: most games get only a handful of reviews")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "05_review_distribution.png", dpi=150); plt.close(fig)

    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    light_table(df).to_parquet(APP_DATA, index=False)
    print("figures ->", config.FIGURES_DIR)
    print("app data ->", APP_DATA, f"{APP_DATA.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
