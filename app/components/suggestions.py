"""Suggested changes, each re-scored by the model, and their wording."""

import streamlit as st

from common import md
from gamesense.game import Game
from gamesense.predictor import Prediction
from gamesense.suggestions import Change, CombinedPlan, ScoredChange

# Title and explanation per change id. Titles can use {old_price} and {new_price}.
TEXT: dict[str, tuple[str, str]] = {
    "more_screenshots": (
        "Show at least 10 screenshots",
        "Cover different areas, mechanics and moods. Steam uses them in search previews and on the store page.",
    ),
    "website": (
        "Put up a game website",
        "A single page with the trailer, a press kit and store links is enough.",
    ),
    "steam_cloud": (
        "Enable Steam Cloud saves",
        "Mostly Steamworks configuration. Lets players continue on another PC or a Steam Deck.",
    ),
    "achievements": (
        "Add around 25 Steam Achievements",
        "Gives players goals and puts the game in friends' activity feeds.",
    ),
    "controller": (
        "Add full controller support",
        "Needed for a Steam Deck Verified rating and for couch players.",
    ),
    "localisation": (
        "Localise into 8 languages",
        "Common first targets are Simplified Chinese, Russian, Spanish, Brazilian Portuguese, German, "
        "French and Japanese. Costs depend heavily on how much text the game has.",
    ),
    "linux": (
        "Ship a native Linux build",
        "Helps with Linux players and Steam Deck. Engines like Unity and Godot can export Linux builds.",
    ),
    "macos": (
        "Ship a macOS build",
        "Requires testing on Apple hardware and notarisation.",
    ),
    "raise_price": (
        "Price at ${new_price:.2f} instead of ${old_price:.2f}",
        "Very cheap games are often read as low effort. Only worth it if the content supports the price.",
    ),
    "lower_price": (
        "Price at ${new_price:.2f} instead of ${old_price:.2f}",
        "Few games from smaller studios sell well above $30.",
    ),
    "online_coop": (
        "Add online co-op",
        "A design change with networking work. Only consider it if it fits the game.",
    ),
    "publisher": (
        "Work with an established publisher",
        "Tested as a publisher with 50+ released games. Publishers take a revenue share in return "
        "for marketing, funding or porting.",
    ),
}


def title(change: Change, game: Game) -> str:
    template = TEXT[change.id][0]
    return template.format(old_price=game.price, new_price=change.updates.get("price", game.price))


def card(s: ScoredChange, game: Game) -> None:
    with st.container(border=True):
        st.markdown(f"**{md(title(s.change, game))}**")
        st.markdown(
            f"Chance of 100+ reviews **{s.success_before:.0%} → {s.success_after:.0%}** "
            f"({s.uplift * 100:+.1f} points)  \nExpected reviews **{(s.reviews_ratio - 1) * 100:+.0f}%**"
        )
        st.caption(f"{s.change.area}, {s.change.effort.lower()} effort. {md(TEXT[s.change.id][1])}")


def plan_summary(plan: CombinedPlan, pred: Prediction, scored: list[ScoredChange], game: Game) -> None:
    by_id = {s.change.id: s.change for s in scored}
    with st.container(border=True):
        st.markdown(f"**All {len(plan.change_ids)} low and medium effort changes together**")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Chance of 100+ reviews",
            f"{plan.success_after:.0%}",
            f"{(plan.success_after - pred.success_chance) * 100:+.1f} points",
        )
        c2.metric("Expected reviews", f"{plan.reviews_after:,.0f}", f"from {pred.reviews_mid:,.0f}", delta_color="off")
        c3.metric("Predicted tier", plan.tier_after, f"from {pred.tier_name}", delta_color="off")
        included = "; ".join(title(by_id[i], game) for i in plan.change_ids)
        st.caption(
            "Includes: " + md(included) + ". The combined game is scored as a whole, "
            "because the effects of separate changes do not simply add up."
        )


def render(game: Game, pred: Prediction, scored: list[ScoredChange], plan: CombinedPlan | None) -> None:
    st.subheader("Suggestions")
    st.markdown(
        "Each suggestion re-runs the model with a single change to your game. A change is listed when it "
        "adds at least 1 point to the chance of 100+ reviews, or at least 15% to expected reviews "
        "without lowering that chance."
    )
    helpful = [s for s in scored if s.helps]
    if not helpful:
        st.markdown(
            "None of the tested changes pass that bar. What holds this game back is mostly outside quick "
            "fixes: genre, studio history or the description itself."
        )

    groups = [
        ("Low and medium effort", [s for s in helpful if s.change.effort != "High"]),
        ("Bigger decisions", [s for s in helpful if s.change.effort == "High"]),
    ]
    for heading, rows in groups:
        if not rows:
            continue
        st.markdown(f"#### {heading}")
        cols = st.columns(2, gap="medium")
        for i, s in enumerate(rows):
            with cols[i % 2]:
                card(s, game)

    if plan:
        plan_summary(plan, pred, scored, game)

    unhelpful = [s for s in scored if not s.helps]
    if unhelpful:
        with st.expander(f"Also tested, with little or negative effect ({len(unhelpful)})"):
            for s in unhelpful:
                st.markdown(
                    f"- {md(title(s.change, game))}: chance {s.success_before:.0%} → {s.success_after:.0%}, "
                    f"expected reviews {(s.reviews_ratio - 1) * 100:+.0f}%"
                )

    st.caption(
        "These effects are associations learned from past launches, not guarantees. For example, games with "
        "controller support also tend to be more polished overall, and the model cannot separate the two."
    )
