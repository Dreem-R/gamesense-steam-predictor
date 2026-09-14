import plotly.graph_objects as go
import streamlit as st

from app.common import (TIER_COLOR, TIER_HEADLINE, VISIBLE_GENRES, load_insights, load_metrics, load_predictor,
                        load_test_predictions)
from app.presets import PRESETS
from src import config
from src.predictor import default_game
from src.reliability import calibration_check, confidence_level, input_warnings, tier_track_record

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CATEGORY_GROUPS = {
    "Multiplayer": ["Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op", "Shared/Split Screen",
                    "Cross-Platform Multiplayer", "MMO", "LAN Co-op", "LAN PvP", "Remote Play Together"],
    "Steam features": ["Steam Achievements", "Steam Cloud", "Steam Leaderboards", "Steam Workshop", "Stats",
                       "In-App Purchases", "Includes level editor", "Captions available"],
    "Controls and VR": ["Full controller support", "Partial Controller Support", "VR Only", "VR Supported"],
}

predictor = load_predictor()


def md(text: str) -> str:
    """Escape dollar signs so Streamlit does not render them as LaTeX."""
    return text.replace("$", r"\$")


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

st.title("Will my game succeed on Steam?")
st.markdown(
    f"Describe the game you are making. The model was trained on "
    f"**{predictor.b['train_rows']:,} Steam games released between 2018 and 2024** and estimates how a game "
    "like yours performs in its first year or two on sale."
)
st.selectbox("Load an example game or start from scratch", list(PRESETS), key="preset", on_change=apply_preset)

left, right = st.columns([1.05, 1], gap="large")

with left:
    t1, t2, t3, t4 = st.tabs(["Game", "Store page", "Features", "Studio"])
    with t1:
        st.text_input("Game title", key="g_name")
        st.multiselect("Genres (as listed on Steam)", VISIBLE_GENRES, key="g_genres")
        st.text_area("Short description (the one-line summary on the store page)", key="g_short", height=80,
                     max_chars=300)
        st.text_area("About this game (full store description)", key="g_about", height=170,
                     help="A text model reads this. A real description gives a more accurate prediction.")
        c1, c2 = st.columns(2)
        c1.toggle("Free to play", key="g_free")
        c1.number_input("Price (USD)", 0.99, 99.99, step=1.0, key="g_price", disabled=st.session_state.g_free)
        c2.select_slider("Release month", MONTHS, key="g_month")
        c2.toggle("Mature content (17+)", key="g_mature")
    with t2:
        st.slider("Screenshots on the store page", 0, 30, key="g_screens")
        st.slider("Languages supported (interface or subtitles)", 1, 30, key="g_langs")
        st.slider("Languages with full voice-over", 0, 15, key="g_audio")
        st.number_input("Steam Achievements", 0, 1000, step=5, key="g_ach")
        st.toggle("Has an official website", key="g_site")
    with t3:
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Windows", key="g_win")
        c2.checkbox("macOS", key="g_mac")
        c3.checkbox("Linux", key="g_linux")
        for group, options in CATEGORY_GROUPS.items():
            st.multiselect(group, options, key=f"g_cat_{group}")
        st.checkbox("Multiplayer only (no single-player mode)", key="g_mp_only")
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

market = load_insights()
test = load_test_predictions()
metrics = load_metrics()

with right:
    if not game["genres"]:
        st.info("Select at least one genre to get a prediction.")
        st.stop()

    pred = predictor.predict(game)
    tier = config.TIER_NAMES[pred.tier]
    market_rate = market.loc[market.year == config.YEAR_MAX, "success"].mean()
    level, level_detail = confidence_level(pred.proba)

    st.markdown(
        f"""<div style="border-radius:10px;padding:16px 20px;background:{TIER_COLOR[tier]}1f;
        border:1px solid {TIER_COLOR[tier]};margin-bottom:12px">
        <div style="font-size:0.85rem;opacity:.75">Prediction for {game['name'] or 'your game'}</div>
        <div style="font-size:2rem;font-weight:700;line-height:1.25">{tier}</div>
        <div style="font-size:1.05rem;font-weight:600">{TIER_HEADLINE[tier]}</div>
        <div style="font-size:0.85rem;opacity:.8;margin-top:4px">{config.TIER_BLURBS[tier]}
        Confidence: {level.lower()}.</div></div>""",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    ratio = pred.success_chance / market_rate
    m1.metric("Chance of 100+ reviews", f"{pred.success_chance:.0%}",
              f"{ratio:.1f}x the average {config.YEAR_MAX} release",
              delta_color="normal" if ratio >= 1 else "inverse")
    m2.metric("Expected reviews", f"{pred.reviews_mid:,.0f}",
              f"likely range {pred.reviews_low:,.0f} to {pred.reviews_high:,.0f}", delta_color="off")
    st.caption(f"At 30 to 60 sales per review, that is roughly {pred.reviews_mid * 30:,.0f} to "
               f"{pred.reviews_mid * 60:,.0f} copies.")

    fig = go.Figure(go.Bar(
        x=pred.proba * 100, y=config.TIER_NAMES, orientation="h",
        marker_color=[TIER_COLOR[t] for t in config.TIER_NAMES],
        text=[f"{p:.0%}" for p in pred.proba], textposition="outside",
    ))
    fig.update_layout(height=200, margin=dict(l=0, r=30, t=30, b=0), title="Probability of each outcome",
                      xaxis=dict(range=[0, 108], showticklabels=False, showgrid=False),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Reliability
st.divider()
st.subheader("How reliable is this prediction?")
st.markdown("These checks compare this prediction with how the model performed on "
            f"**{metrics['rows_test']:,} games released in 2024**, which were held out from training.")

track = tier_track_record(test, pred.tier)
calib = calibration_check(test, pred.success_chance)
r1, r2, r3 = st.columns(3)
with r1:
    with st.container(border=True):
        st.metric("Confidence", level)
        st.caption(md(level_detail) + " When probabilities are spread across tiers, treat the tier label "
                   "as a rough guide and look at the chance of 100+ reviews instead.")
with r2:
    with st.container(border=True):
        st.metric(f"Past accuracy for {tier} predictions", f"{track['exact']:.0%}")
        st.caption(f"Of {track['games']:,} games from 2024 that the model placed in {tier}, "
                   f"{track['exact']:.0%} ended up there and {track['within_one']:.0%} were within one tier.")
with r3:
    with st.container(border=True):
        if calib["games"] >= 30:
            st.metric("Did similar predictions come true?", f"{calib['actual_rate']:.0%}")
            st.caption(f"{calib['games']:,} games from 2024 were given a {calib['low']:.0%} to "
                       f"{calib['high']:.0%} chance of 100+ reviews. {calib['actual_rate']:.0%} of them "
                       "actually got there.")
        else:
            st.metric("Did similar predictions come true?", "Too few cases")
            st.caption("Fewer than 30 games from 2024 were given a chance this close, so there is not "
                       "enough data to check it.")

coverage = metrics["review_range_test_2024"]["interval_80_coverage"]
st.caption(f"The review range is the model's 10th to 90th percentile estimate. For 2024 games, the actual review "
           f"count fell inside that range {coverage:.0%} of the time, so the real number can land outside it.")

for note in input_warnings(game, market):
    st.warning(md(note))

k1, k2 = st.columns(2)
with k1:
    st.markdown("**What the model uses**")
    st.markdown(
        "- Genres, price, platforms and Steam features\n"
        "- Store page setup: screenshots, languages, achievements, description\n"
        "- The studio's and publisher's previous releases\n"
        "- Patterns from 69,000 real launches"
    )
with k2:
    st.markdown("**What the model cannot see**")
    st.markdown(
        "- Whether the game is fun, polished or original\n"
        "- Trailer quality, capsule art and wishlists before launch\n"
        "- Marketing, press, festivals and streamer coverage\n"
        "- Launch timing against big releases and Steam sales"
    )
st.info("Best use: change one thing at a time and compare the results, instead of treating a single number "
        "as a forecast. A low score does not mean the idea is bad. It means games with a similar store setup "
        "usually struggled, and the parts the model cannot see need to do more of the work.")

# Explanation and suggestions
st.divider()
st.subheader("What is driving the prediction")
exp = predictor.explain(game, top_k=8).iloc[::-1]
colors = ["#66a182" if e > 0 else "#d1495b" for e in exp.effect]
labels = [f"{l} ({v})" for l, v in zip(exp.label, exp.value)]
fig = go.Figure(go.Bar(x=exp.effect, y=labels, orientation="h", marker_color=colors,
                       text=[f"x{m:.2f}" for m in exp.multiplier], textposition="auto",
                       hovertemplate="%{y}<br>reviews %{text}<extra></extra>"))
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                  xaxis_title="lowers expected reviews  |  raises expected reviews", xaxis_zeroline=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption("Each bar shows how much a factor moves the expected review count compared with an average game "
           "(SHAP values from the review model). x1.50 means 50% more reviews, x0.70 means 30% fewer.")


def suggestion_card(r):
    with st.container(border=True):
        st.markdown(f"**{md(r.change)}**")
        st.markdown(
            f"Chance of 100+ reviews **{r.success_before:.0%} → {r.success_after:.0%}** "
            f"({r.uplift * 100:+.1f} points)  \nExpected reviews **{(r.reviews_ratio - 1) * 100:+.0f}%**"
        )
        st.caption(f"{r.area}, {r.effort.lower()} effort. {md(r.detail)}")


st.divider()
st.subheader("Suggestions")
st.markdown("Each suggestion re-runs the model with a single change to your game. A change is listed when it "
            "adds at least 1 point to the chance of 100+ reviews, or at least 15% to expected reviews "
            "without lowering that chance.")
table, plan = predictor.what_if(game)
helpful = table[table.helps]

if helpful.empty:
    st.markdown("None of the tested changes pass that bar. What holds this game back is mostly outside quick "
                "fixes: genre, studio history or the description itself.")

for title, rows in [("Low and medium effort", helpful[helpful.effort != "High"]),
                    ("Bigger decisions", helpful[helpful.effort == "High"])]:
    if rows.empty:
        continue
    st.markdown(f"#### {title}")
    cols = st.columns(2, gap="medium")
    for i, (_, r) in enumerate(rows.iterrows()):
        with cols[i % 2]:
            suggestion_card(r)

if plan:
    with st.container(border=True):
        st.markdown(f"**All {len(plan['changes'])} low and medium effort changes together**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chance of 100+ reviews", f"{plan['success_after']:.0%}",
                  f"{(plan['success_after'] - pred.success_chance) * 100:+.1f} points")
        c2.metric("Expected reviews", f"{plan['reviews_after']:,.0f}",
                  f"from {pred.reviews_mid:,.0f}", delta_color="off")
        c3.metric("Predicted tier", plan["tier_after"], f"from {tier}", delta_color="off")
        st.caption("Includes: " + md("; ".join(plan["changes"])) + ". The combined game is scored as a whole, "
                   "because the effects of separate changes do not simply add up.")

unhelpful = table[~table.helps]
if not unhelpful.empty:
    with st.expander(f"Also tested, with little or negative effect ({len(unhelpful)})"):
        for _, r in unhelpful.iterrows():
            st.markdown(f"- {md(r.change)}: chance {r.success_before:.0%} → {r.success_after:.0%}, "
                        f"expected reviews {(r.reviews_ratio - 1) * 100:+.0f}%")

st.caption("These effects are associations learned from past launches, not guarantees. For example, games with "
           "controller support also tend to be more polished overall, and the model cannot separate the two.")
