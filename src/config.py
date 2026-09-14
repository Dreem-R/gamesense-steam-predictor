"""Paths, feature vocabularies and target definition."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "steam_games.parquet"
PROCESSED_PATH = ROOT / "data" / "processed" / "games_clean.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Every game in this window had at least a year on sale before the snapshot.
YEAR_MIN, YEAR_MAX = 2018, 2024
TRAIN_YEARS = (2018, 2022)
VALID_YEARS = (2023, 2023)
TEST_YEARS = (2024, 2024)

# Tiers by total reviews. Boxleiter estimate: roughly 30-60 copies sold per review.
TIER_BINS = [-1, 9, 99, 999, float("inf")]
TIER_NAMES = ["Flop", "Niche", "Success", "Hit"]
TIER_BLURBS = {
    "Flop": "Fewer than 10 reviews (roughly under ~500 copies sold).",
    "Niche": "10-99 reviews (roughly 500-5,000 copies).",
    "Success": "100-999 reviews (roughly 5,000-50,000 copies).",
    "Hit": "1,000+ reviews (roughly 50,000+ copies).",
}

NON_GAME_GENRES = {
    "Utilities", "Design & Illustration", "Animation & Modeling", "Education",
    "Video Production", "Game Development", "Audio Production", "Software Training",
    "Photo Editing", "Web Publishing", "Accounting", "Movie", "Documentary",
    "Episodic", "Short", "Tutorial", "360 Video",
}

GENRES = [
    "Indie", "Casual", "Action", "Adventure", "Simulation", "Strategy", "RPG",
    "Free To Play", "Early Access", "Sports", "Racing", "Massively Multiplayer",
    "Violent", "Gore", "Nudity", "Sexual Content",
]

# Excluded as leaky: Steam Trading Cards and Profile Features Limited depend on
# sales, Family Sharing and Steam Timeline were added to most games after launch.
CATEGORIES = [
    "Single-player", "Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op",
    "Shared/Split Screen", "Cross-Platform Multiplayer", "MMO", "Steam Achievements",
    "Steam Cloud", "Full controller support", "Partial Controller Support",
    "Steam Leaderboards", "Remote Play Together", "Steam Workshop", "In-App Purchases",
    "VR Only", "VR Supported", "Includes level editor", "Stats", "Captions available",
    "LAN Co-op", "LAN PvP",
]

TARGET_REVIEWS = "total_reviews"
TARGET_TIER = "tier"
