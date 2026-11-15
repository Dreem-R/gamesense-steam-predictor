"""Paths, feature vocabularies and target definition."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "steam_games.parquet"
PROCESSED_PATH = ROOT / "data" / "processed" / "games_clean.parquet"
FORWARD_PATH = ROOT / "data" / "processed" / "games_forward.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DATA_PATH = ROOT / "app" / "data" / "insights.parquet"
MODEL_PATH = MODELS_DIR / "gamesense.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test2024_predictions.parquet"
FORWARD_REPORT_PATH = REPORTS_DIR / "forward_test.json"

RANDOM_STATE = 42

# The tiers come from SteamSpy's review counts, which stop updating in mid-2025 in this dataset
# (see the forward test). Later release years have no usable tiers.
YEAR_MIN, YEAR_MAX = 2018, 2024
TRAIN_YEARS = (2018, 2022)
VALID_YEARS = (2023, 2023)
TEST_YEARS = (2024, 2024)

# The first release year after the window. The shipped model never sees it, so it is used to check
# that model on later launches. Games with less time on sale than this have not collected their reviews yet.
FORWARD_YEAR = YEAR_MAX + 1
MIN_MONTHS_ON_SALE = 12

# Tiers by total reviews. Boxleiter estimate: roughly 30-60 copies sold per review.
TIER_BINS = [-1, 9, 99, 999, float("inf")]
TIER_NAMES = ["Flop", "Niche", "Success", "Hit"]
TIER_BLURBS = {
    "Flop": "Fewer than 10 reviews, under about 500 copies sold.",
    "Niche": "10 to 99 reviews, about 500 to 5,000 copies sold.",
    "Success": "100 to 999 reviews, about 5,000 to 50,000 copies sold.",
    "Hit": "1,000 or more reviews, about 50,000 copies sold or more.",
}

NON_GAME_GENRES = {
    "Utilities",
    "Design & Illustration",
    "Animation & Modeling",
    "Education",
    "Video Production",
    "Game Development",
    "Audio Production",
    "Software Training",
    "Photo Editing",
    "Web Publishing",
    "Accounting",
    "Movie",
    "Documentary",
    "Episodic",
    "Short",
    "Tutorial",
    "360 Video",
}

GENRES = [
    "Indie",
    "Casual",
    "Action",
    "Adventure",
    "Simulation",
    "Strategy",
    "RPG",
    "Free To Play",
    "Early Access",
    "Sports",
    "Racing",
    "Massively Multiplayer",
    "Violent",
    "Gore",
    "Nudity",
    "Sexual Content",
]

# Excluded as leaky: Steam Trading Cards and Profile Features Limited depend on
# sales, Family Sharing and Steam Timeline were added to most games after launch.
CATEGORIES = [
    "Single-player",
    "Multi-player",
    "PvP",
    "Online PvP",
    "Co-op",
    "Online Co-op",
    "Shared/Split Screen",
    "Cross-Platform Multiplayer",
    "MMO",
    "Steam Achievements",
    "Steam Cloud",
    "Full controller support",
    "Partial Controller Support",
    "Steam Leaderboards",
    "Remote Play Together",
    "Steam Workshop",
    "In-App Purchases",
    "VR Only",
    "VR Supported",
    "Includes level editor",
    "Stats",
    "Captions available",
    "LAN Co-op",
    "LAN PvP",
]

TARGET_REVIEWS = "total_reviews"
TARGET_TIER = "tier"
