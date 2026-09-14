import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import HIDDEN_GENRES, TIER_COLOR, VISIBLE_GENRES, load_insights
from src import config

df = load_insights()
df = df[~df.genres.apply(lambda g: bool(HIDDEN_GENRES & set(g)))]

st.title("📊 Steam market insights")
st.markdown(f"What **{len(df):,} Steam games released 2018-2024** tell us about success. "
            "\"Success\" here means reaching **100+ reviews** (roughly 5,000+ copies sold).")

genres = st.multiselect("Filter by genre (games must have all selected genres)", VISIBLE_GENRES,
                        placeholder="All genres")
d = df[df.genres.apply(lambda g: set(genres) <= set(g))] if genres else df
if len(d) < 50:
    st.warning("Fewer than 50 games match this filter. Try fewer genres.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Games", f"{len(d):,}")
k2.metric("Reach 100+ reviews", f"{d.success.mean():.1%}")
k3.metric("Become a Hit (1k+)", f"{(d.tier == 3).mean():.1%}")
k4.metric("Median reviews", f"{d.total_reviews.median():.0f}")


def rate_chart(frame, col, bins, labels, title):
    s = (frame.assign(bin=pd.cut(frame[col], bins, labels=labels))
         .groupby("bin", observed=True).agg(rate=("success", "mean"), games=("success", "size")).reset_index())
    fig = px.bar(s, x="bin", y="rate", text=s.rate.map("{:.0%}".format), hover_data={"games": True},
                 title=title, color_discrete_sequence=["#2e86ab"])
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), yaxis_tickformat=".0%",
                      xaxis_title=None, yaxis_title="success rate")
    return fig


c1, c2 = st.columns(2)
c1.plotly_chart(rate_chart(d, "price", [-0.01, 0, 4.99, 9.99, 19.99, 29.99, 100],
                           ["Free", "<$5", "$5-10", "$10-20", "$20-30", "$30+"], "Price"), use_container_width=True)
c2.plotly_chart(rate_chart(d, "n_languages", [-1, 1, 3, 7, 12, 100],
                           ["1", "2-3", "4-7", "8-12", "13+"], "Languages supported"), use_container_width=True)
c1.plotly_chart(rate_chart(d, "n_screenshots", [-1, 4, 7, 10, 15, 500],
                           ["0-4", "5-7", "8-10", "11-15", "16+"], "Store-page screenshots"), use_container_width=True)
c2.plotly_chart(rate_chart(d, "dev_prior_games", [-1, 0, 1, 3, 10, 1000],
                           ["First game", "2nd", "3rd-4th", "5th-11th", "12th+"], "Studio experience"),
                use_container_width=True)

yr = d.groupby("year").agg(games=("success", "size"), rate=("success", "mean")).reset_index()
fig = px.bar(yr, x="year", y="games", title="Releases per year (bars) and success rate (line)",
             color_discrete_sequence=["#c9d6df"])
fig.add_scatter(x=yr.year, y=yr.rate, yaxis="y2", mode="lines+markers", name="success rate",
                line=dict(color="#d1495b", width=3))
fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                  yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
st.plotly_chart(fig, use_container_width=True)
st.caption("Recent years have lower rates partly because those games have had less time to collect reviews.")

st.subheader("Which Steam features go with success?")
cats = d[["categories", "success"]].explode("categories")
cs = cats.groupby("categories").agg(games=("success", "size"), rate=("success", "mean")).reset_index()
cs = cs[(cs.games >= 100) & cs.categories.isin(config.CATEGORIES)].sort_values("rate")
fig = px.bar(cs, x="rate", y="categories", orientation="h", hover_data={"games": True},
             color_discrete_sequence=["#66a182"])
fig.add_vline(x=d.success.mean(), line_dash="dash", annotation_text="average")
fig.update_layout(height=max(300, 22 * len(cs)), margin=dict(l=0, r=0, t=10, b=0),
                  xaxis_tickformat=".0%", xaxis_title="success rate", yaxis_title=None)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Outcome mix")
mix = d.tier.map(dict(enumerate(config.TIER_NAMES))).value_counts(normalize=True).reindex(config.TIER_NAMES).reset_index()
mix.columns = ["tier", "share"]
fig = px.pie(mix, names="tier", values="share", color="tier", color_discrete_map=TIER_COLOR, hole=0.5)
fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Most-reviewed games in this selection"):
    top = d.nlargest(15, "total_reviews")[["name", "year", "price", "total_reviews", "positive_ratio"]]
    st.dataframe(top.rename(columns={"total_reviews": "reviews", "positive_ratio": "% positive"})
                 .style.format({"price": "${:.2f}", "reviews": "{:,}", "% positive": "{:.0%}"}),
                 hide_index=True, use_container_width=True)
