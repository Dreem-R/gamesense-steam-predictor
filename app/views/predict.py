import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.common import (TIER_COLOR, TIER_EMOJI, TIER_HEADLINE, VISIBLE_GENRES, load_insights, load_metrics,
                        load_predictor)
from app.presets import PRESETS
from src import config
from src.predictor import MIN_DESCRIPTION_WORDS, default_game

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CATEGORY_GROUPS = {
    "Multiplayer": ["Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op", "Shared/Split Screen",
                    "Cross-Platform Multiplayer", "MMO", "LAN Co-op", "LAN PvP", "Remote Play Together"],
    "Steam features": ["Steam Achievements", "Steam Cloud", "Steam Leaderboards", "Steam Workshop", "Stats",
                       "In-App Purchases", "Includes level editor", "Captions available"],
    "Controls & VR": ["Full controller support", "Partial Controller Support", "VR Only", "VR Supported"],
}

predictor = load_predictor()


def apply_preset():
    g = default_game(**PRESETS[st.session_state.preset])
    st.session_state.update({
        "g_name": g["name"], "g_short": g["short_description"], "g_about": g["about"],
        "g_genres": g["genres"], "g_free": g["price"] == 0, "g_price": max(g["price"], 0.99),
        "g_month": MONTHS[g["month"] - 1], "g_screens": g["n_screenshots"], "g_langs": g["n_languages"],
        "g_audio": g["n_audio_languages"], "g_site": bool(g["has_website"]), "g_ach": g["achievements"],
        "g_win": bool(g["windows"]), "g_mac": bool(g["mac"]), "g_linux": bool(g["linux"]),
        "g_mature": g["required_age"] >= 17, "g_dev_games": g["dev_prior_games"],
        "g_dev_best": g["dev_prior_best_reviews"], "g_selfpub": bool(g["self_published"]),
        "g_pub_games": g["pub_prior_games"], "g_pub_best": g["pub_prior_best_reviews"],
    })
    for group, options in CATEGORY_GROUPS.items():
        st.session_state[f"g_cat_{group}"] = [c for c in g["categories"] if c in options]


if "g_name" not in st.session_state:
    st.session_state.preset = list(PRESETS)[1]
    apply_preset()

# ---------------------------------------------------------------------------
st.title("🎮 Will my game succeed on Steam?")
st.markdown(
    f"Describe the game you're making. **GameSense** was trained on "
    f"**{predictor.b['train_rows']:,} real Steam releases (2018-2024)** and predicts how it's likely to do "
    "in its first year or two on sale, why, and what you could change."
)
st.selectbox("Try an example pitch, or start from scratch", list(PRESETS), key="preset", on_change=apply_preset)

left, right = st.columns([1.05, 1], gap="large")

with left:
    t1, t2, t3, t4 = st.tabs(["🎲 The game", "🛒 Store page", "⚙️ Features", "🏢 Studio"])
    with t1:
        st.text_input("Game title", key="g_name")
        st.multiselect("Genres (as listed on Steam)", VISIBLE_GENRES, key="g_genres")
        st.text_area("Short description (the one-liner under the trailer)", key="g_short", height=80, max_chars=300)
        st.text_area("About this game (store description)", key="g_about", height=170,
                     help="An NLP model reads this. More detail means a more accurate prediction.")
        c1, c2 = st.columns(2)
        c1.toggle("Free to play", key="g_free")
        c1.number_input("Price (USD)", 0.99, 99.99, step=1.0, key="g_price", disabled=st.session_state.g_free)
        c2.select_slider("Release month", MONTHS, key="g_month")
        c2.toggle("Mature content (17+)", key="g_mature")
    with t2:
        st.slider("Screenshots on the store page", 0, 30, key="g_screens")
        st.slider("Languages supported (interface/subtitles)", 1, 30, key="g_langs")
        st.slider("Languages with full voice-over", 0, 15, key="g_audio")
        st.number_input("Steam Achievements", 0, 1000, step=5, key="g_ach")
        st.toggle("Has an official website", key="g_site")
    with t3:
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Windows", key="g_win")
        c2.checkbox("macOS", key="g_mac")
        c3.checkbox("Linux / Steam Deck", key="g_linux")
        for group, options in CATEGORY_GROUPS.items():
            st.multiselect(group, options, key=f"g_cat_{group}")
        st.caption("Single-player is assumed unless the game is multiplayer-only.")
        st.checkbox("Multiplayer only (no single-player)", key="g_mp_only")
    with t4:
        st.number_input("Games this studio has released on Steam before", 0, 100, key="g_dev_games")
        st.number_input("Reviews on the studio's best previous game", 0, 1_000_000, step=50, key="g_dev_best",
                        disabled=st.session_state.g_dev_games == 0)
        st.toggle("Self-published", key="g_selfpub")
        if not st.session_state.g_selfpub:
            st.number_input("Games the publisher has released", 0, 2000, key="g_pub_games")
            st.number_input("Reviews on the publisher's best game", 0, 5_000_000, step=500, key="g_pub_best")

s = st.session_state
categories = [c for g in CATEGORY_GROUPS for c in s[f"g_cat_{g}"]]
if not s.get("g_mp_only"):
    categories.append("Single-player")
about = s.g_about or ""
game = default_game(
    name=s.g_name, short_description=s.g_short or "", about=about, about_words=len(about.split()),
    genres=s.g_genres, categories=sorted(set(categories)), price=0.0 if s.g_free else float(s.g_price),
    month=MONTHS.index(s.g_month) + 1, required_age=17 if s.g_mature else 0,
    n_screenshots=s.g_screens, n_languages=s.g_langs, n_audio_languages=s.g_audio, achievements=s.g_ach,
    has_website=int(s.g_site), windows=int(s.g_win), mac=int(s.g_mac), linux=int(s.g_linux),
    dev_prior_games=s.g_dev_games, dev_prior_best_reviews=s.g_dev_best if s.g_dev_games else 0,
    self_published=int(s.g_selfpub),
    pub_prior_games=0 if s.g_selfpub else s.get("g_pub_games", 0),
    pub_prior_best_reviews=0 if s.g_selfpub else s.get("g_pub_best", 0),
)
if game["self_published"]:
    game["pub_prior_games"], game["pub_prior_best_reviews"] = game["dev_prior_games"], game["dev_prior_best_reviews"]

# ---------------------------------------------------------------------------
with right:
    if not game["genres"]:
        st.info("Pick at least one genre to get a prediction.")
        st.stop()

    pred = predictor.predict(game)
    tier = config.TIER_NAMES[pred.tier]
    market = load_insights()
    market_rate = market.loc[market.year == config.YEAR_MAX, "success"].mean()

    st.markdown(
        f"""<div style="border-radius:14px;padding:18px 22px;background:{TIER_COLOR[tier]}1f;
        border:2px solid {TIER_COLOR[tier]};margin-bottom:10px">
        <div style="font-size:0.9rem;opacity:.75">Prediction for <b>{game['name'] or 'your game'}</b></div>
        <div style="font-size:2.1rem;font-weight:800;line-height:1.2">{TIER_EMOJI[tier]} {tier}</div>
        <div style="font-size:1.1rem;font-weight:600">{TIER_HEADLINE[tier]}</div>
        <div style="font-size:0.9rem;opacity:.8;margin-top:4px">{config.TIER_BLURBS[tier]}</div></div>""",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m1.metric("Chance of 100+ reviews", f"{pred.success_chance:.0%}",
              f"{pred.success_chance / market_rate:.1f}× the average {config.YEAR_MAX} release",
              delta_color="normal" if pred.success_chance >= market_rate else "inverse")
    m2.metric("Expected reviews", f"~{pred.reviews_mid:,.0f}",
              f"likely range {pred.reviews_low:,.0f} – {pred.reviews_high:,.0f}", delta_color="off")
    st.caption(f"Rule of thumb: 30-60 sales per review ≈ **{pred.reviews_mid * 30:,.0f} – "
               f"{pred.reviews_mid * 60:,.0f} copies**.")
    if not pred.used_text:
        st.warning(f"Add a description of at least {MIN_DESCRIPTION_WORDS} words so the NLP part of the model can read it.")

    fig = go.Figure(go.Bar(
        x=pred.proba * 100, y=[f"{TIER_EMOJI[t]} {t}" for t in config.TIER_NAMES], orientation="h",
        marker_color=[TIER_COLOR[t] for t in config.TIER_NAMES],
        text=[f"{p:.0%}" for p in pred.proba], textposition="outside",
    ))
    fig.update_layout(height=210, margin=dict(l=0, r=30, t=30, b=0), title="Probability of each outcome",
                      xaxis=dict(range=[0, 105], showticklabels=False, showgrid=False),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()
c_why, c_fix = st.columns(2, gap="large")
with c_why:
    st.subheader("🔍 Why the model thinks so")
    exp = predictor.explain(game, top_k=8).iloc[::-1]
    colors = ["#66a182" if e > 0 else "#d1495b" for e in exp.effect]
    labels = [f"{l} ({v})" for l, v in zip(exp.label, exp.value)]
    fig = go.Figure(go.Bar(x=exp.effect, y=labels, orientation="h", marker_color=colors,
                           text=[f"×{m:.2f}" for m in exp.multiplier], textposition="auto",
                           hovertemplate="%{y}<br>reviews ×%{text}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="← fewer reviews  |  more reviews →", xaxis_zeroline=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Each bar is that factor's push on expected reviews compared with an average game "
               "(SHAP-style contributions). ×1.5 = 50% more reviews.")

with c_fix:
    st.subheader("🛠️ Quick wins to try")
    wi = predictor.what_if(game)
    wins = wi[wi.uplift > 0.005].head(6)
    if wins.empty:
        st.success("No obvious quick wins, your store setup already covers the basics.")
    for _, r in wins.iterrows():
        change = r.change.replace("$", r"\$")
        st.markdown(
            f"**{change}**  \n"
            f"<span style='color:#66a182;font-weight:700'>+{r.uplift * 100:.1f} pts</span> "
            f"→ {r.success_chance:.0%} chance of 100+ reviews",
            unsafe_allow_html=True,
        )
        st.progress(float(np.clip(r.success_chance, 0, 1)))
    st.caption("The model re-scores your game with each single change. It shows patterns in the data, "
               "not a guarantee that the change alone causes success.")

test_scores = load_metrics()["test_2024"]["GameSense ensemble"]
with st.expander("⚠️ How much should I trust this?"):
    st.markdown(
        f"- On **2024 releases the model never saw**, it predicted the exact tier for "
        f"{test_scores['accuracy']:.0%} of games and was within one tier for {test_scores['within_1_tier']:.0%}. "
        f"See **How it works** for all the numbers.\n"
        "- It only knows what's on the store page and the studio's track record. It can't see how *fun* your "
        "game is, your trailer, wishlists, marketing, or a streamer picking it up.\n"
        "- Treat it as a reality check on your positioning, not a verdict on your idea."
    )
