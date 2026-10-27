"""The game description form. Widget values live in st.session_state under g_* keys."""

import streamlit as st

from common import VISIBLE_GENRES
from gamesense.features import slug
from gamesense.game import Game
from gamesense.presets import PRESETS

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CATEGORY_GROUPS = {
    "Multiplayer": [
        "Multi-player",
        "PvP",
        "Online PvP",
        "Co-op",
        "Online Co-op",
        "Shared/Split Screen",
        "Cross-Platform Multiplayer",
        "MMO",
        "LAN Co-op",
        "LAN PvP",
        "Remote Play Together",
    ],
    "Steam features": [
        "Steam Achievements",
        "Steam Cloud",
        "Steam Leaderboards",
        "Steam Workshop",
        "Stats",
        "In-App Purchases",
        "Includes level editor",
        "Captions available",
    ],
    "Controls and VR": ["Full controller support", "Partial Controller Support", "VR Only", "VR Supported"],
}
DEFAULT_PRESET = "Cozy farming sim with online co-op"


def init_state() -> None:
    if "g_name" in st.session_state:
        return
    # ?example=story-driven-visual-novel opens the page with that example loaded
    by_slug = {slug(name).replace("_", "-"): name for name in PRESETS}
    st.session_state.preset = by_slug.get(st.query_params.get("example", ""), DEFAULT_PRESET)
    load_preset()


def load_preset() -> None:
    g = PRESETS[st.session_state.preset]
    st.session_state.update(
        {
            "g_name": g.name,
            "g_short": g.short_description,
            "g_about": g.about,
            "g_genres": g.genres,
            "g_free": g.price == 0,
            "g_price": max(g.price, 0.99),
            "g_month": MONTHS[g.month - 1],
            "g_mature": g.required_age >= 17,
            "g_screens": g.n_screenshots,
            "g_langs": g.n_languages,
            "g_audio": g.n_audio_languages,
            "g_site": g.has_website,
            "g_ach": g.achievements,
            "g_win": g.windows,
            "g_mac": g.mac,
            "g_linux": g.linux,
            "g_mp_only": "Single-player" not in g.categories,
            "g_dev_games": g.dev_prior_games,
            "g_dev_best": g.dev_prior_best_reviews,
            "g_selfpub": g.self_published,
            "g_pub_games": g.pub_prior_games,
            "g_pub_best": g.pub_prior_best_reviews,
        }
    )
    for group, options in CATEGORY_GROUPS.items():
        st.session_state[f"g_cat_{group}"] = [c for c in g.categories if c in options]


def preset_picker() -> None:
    st.selectbox("Load an example game or start from scratch", list(PRESETS), key="preset", on_change=load_preset)


def render() -> None:
    game_tab, store_tab, features_tab, studio_tab = st.tabs(["Game", "Store page", "Features", "Studio"])
    with game_tab:
        st.text_input("Game title", key="g_name")
        st.multiselect("Genres (as listed on Steam)", VISIBLE_GENRES, key="g_genres")
        st.text_area(
            "Short description (the one-line summary on the store page)", key="g_short", height=80, max_chars=300
        )
        st.text_area(
            "About this game (full store description)",
            key="g_about",
            height=170,
            help="A text model reads this. A real description gives a more accurate prediction.",
        )
        c1, c2 = st.columns(2)
        c1.toggle("Free to play", key="g_free")
        c1.number_input("Price (USD)", 0.99, 99.99, step=1.0, key="g_price", disabled=st.session_state.g_free)
        c2.select_slider("Release month", MONTHS, key="g_month")
        c2.toggle("Mature content (17+)", key="g_mature")
    with store_tab:
        st.slider("Screenshots on the store page", 0, 30, key="g_screens")
        st.slider("Languages supported (interface or subtitles)", 1, 30, key="g_langs")
        st.slider("Languages with full voice-over", 0, 15, key="g_audio")
        st.number_input("Steam Achievements", 0, 1000, step=5, key="g_ach")
        st.toggle("Has an official website", key="g_site")
    with features_tab:
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Windows", key="g_win")
        c2.checkbox("macOS", key="g_mac")
        c3.checkbox("Linux", key="g_linux")
        for group, options in CATEGORY_GROUPS.items():
            st.multiselect(group, options, key=f"g_cat_{group}")
        st.checkbox("Multiplayer only (no single-player mode)", key="g_mp_only")
    with studio_tab:
        st.number_input("Games this studio has released on Steam before", 0, 100, key="g_dev_games")
        st.number_input(
            "Reviews on the studio's best previous game",
            0,
            1_000_000,
            step=50,
            key="g_dev_best",
            disabled=st.session_state.g_dev_games == 0,
        )
        st.toggle("Self-published", key="g_selfpub")
        if not st.session_state.g_selfpub:
            st.number_input("Games the publisher has released", 0, 2000, key="g_pub_games")
            st.number_input("Reviews on the publisher's best game", 0, 5_000_000, step=500, key="g_pub_best")


def current_game() -> Game | None:
    """The game described by the form, or None until at least one genre is picked."""
    s = st.session_state
    if not s.g_genres:
        return None
    categories = [c for group in CATEGORY_GROUPS for c in s[f"g_cat_{group}"]]
    if not s.get("g_mp_only"):
        categories.append("Single-player")
    return Game(
        name=s.g_name,
        short_description=s.g_short or "",
        about=s.g_about or "",
        genres=s.g_genres,
        categories=sorted(set(categories)),
        price=0.0 if s.g_free else float(s.g_price),
        month=MONTHS.index(s.g_month) + 1,
        required_age=17 if s.g_mature else 0,
        n_screenshots=s.g_screens,
        n_languages=s.g_langs,
        n_audio_languages=s.g_audio,
        achievements=s.g_ach,
        has_website=s.g_site,
        windows=s.g_win,
        mac=s.g_mac,
        linux=s.g_linux,
        dev_prior_games=s.g_dev_games,
        dev_prior_best_reviews=s.g_dev_best,
        self_published=s.g_selfpub,
        pub_prior_games=s.get("g_pub_games", 0),
        pub_prior_best_reviews=s.get("g_pub_best", 0),
    )
