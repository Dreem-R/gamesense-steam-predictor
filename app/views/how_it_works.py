import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import HIDDEN_GENRES, load_metrics, load_predictor
from src import config
from src.features import FRIENDLY_NAMES, slug

metrics = load_metrics()
predictor = load_predictor()

st.title("How it works")

st.markdown(f"""
**The question:** from things a developer knows *before launch*, can we predict how a game will do on Steam?

**The data:** the public *Steam Games Dataset* (FronkonGames, Hugging Face), about 124k apps. After cleaning,
**{metrics['rows_total']:,} games released 2018-2024** remain. Removed rows: empty store pages, software
(not games), untracked apps and prices above \\$100.

**The target:** total Steam reviews, the standard public stand-in for sales, grouped into four tiers:
""")
st.dataframe(pd.DataFrame({"Tier": config.TIER_NAMES, "Meaning": [config.TIER_BLURBS[t] for t in config.TIER_NAMES]}),
             hide_index=True, use_container_width=True)

st.subheader("Pipeline")
st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f7", fontname="Helvetica", fontsize=11];
  raw [label="124k Steam apps"]; clean [label="Clean + filter\\n69k games"];
  feat [label="65 pre-launch features\\nprice, genres, platforms,\\nlanguages, store page,\\nstudio track record"];
  text [label="Store description\\nTF-IDF + Ridge\\n(NLP score, out-of-fold)"];
  xgb [label="XGBoost\\n(Optuna-tuned)"]; lgb [label="LightGBM\\n(Optuna-tuned)"];
  ens [label="Weighted ensemble\\n→ tier probabilities", fillcolor="#d8ecdf"];
  q [label="Quantile LightGBM\\n→ review range\\n+ explanations", fillcolor="#d8ecdf"];
  raw -> clean -> feat; clean -> text -> feat; feat -> xgb -> ens; feat -> lgb -> ens; feat -> q;
}
""")

st.markdown("""
**Leakage controls:**
- **Time-based split:** trained on 2018-2022, tuned on 2023, **tested once on 2024 releases**. That mirrors
  predicting a game that hasn't launched yet.
- **No post-launch signals:** user tags, DLC count, Metacritic, playtime, Steam Trading Cards (Valve grants
  them based on sales) and "Family Sharing" were all excluded.
- **Description scrubbing:** phrases added after a game succeeds ("award-winning", "1M players",
  "Deluxe Edition", "update") are removed before the NLP model reads the text.
- **Studio history** only counts games released *before* each game.
""")

st.subheader("Results on 2024 releases (never seen during training)")
test = pd.DataFrame(metrics["test_2024"]).T
show = test[["accuracy", "macro_f1", "qwk", "within_1_tier", "auc_ovr", "auc_success_plus"]].rename(columns={
    "accuracy": "Accuracy", "macro_f1": "Macro F1", "qwk": "Quadratic κ", "within_1_tier": "Within 1 tier",
    "auc_ovr": "ROC-AUC (macro)", "auc_success_plus": "AUC: 100+ reviews?",
})
st.dataframe(show.style.format("{:.3f}").highlight_max(axis=0, color="#d8ecdf"), use_container_width=True)
best = metrics["test_2024"]["GameSense ensemble"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC: will it get 100+ reviews?", f"{best['auc_success_plus']:.3f}")
c2.metric("Right tier or one off", f"{best['within_1_tier']:.1%}")
c3.metric("Quadratic weighted κ", f"{best['qwk']:.3f}")
baseline_f1 = metrics["test_2024"]["Baseline: majority class"]["macro_f1"]
c4.metric(f"Macro F1 (baseline {baseline_f1:.2f})", f"{best['macro_f1']:.3f}")

cm = pd.DataFrame(metrics["confusion_matrix_test_2024"], index=config.TIER_NAMES, columns=config.TIER_NAMES)
cm_pct = cm.div(cm.sum(axis=1), axis=0)
fig = px.imshow(cm_pct, text_auto=".0%", color_continuous_scale="Blues", aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="share of row"))
fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="Confusion matrix (row %)")
st.plotly_chart(fig, use_container_width=True)

r = metrics["review_range_test_2024"]
st.markdown(f"""
**Review-count estimate** (quantile regression): the median estimate lands within 3x of the true review count
for **{r['within_x3_of_actual']:.0%}** of 2024 games. The 10th-90th percentile range contains the true value
for **{r['interval_80_coverage']:.0%}** of games (Spearman ρ = {r['spearman']:.2f}).
""")

st.subheader("What drives the prediction overall")
imp = pd.Series(predictor.b["lgb"].booster_.feature_importance("gain"), index=predictor.cols)
imp = imp / imp.sum()
imp = imp[[c for c in imp.index if c not in {f"genre_{slug(g)}" for g in HIDDEN_GENRES}]]
imp = imp.sort_values(ascending=False).head(15).iloc[::-1]
fig = px.bar(x=imp.values, y=[FRIENDLY_NAMES.get(i, i) for i in imp.index], orientation="h",
             color_discrete_sequence=["#2e86ab"])
fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="share of total gain",
                  yaxis_title=None, xaxis_tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Limitations")
st.markdown("""
- Reviews measure **reach**, not quality or profit. A \\$3 game with 1,000 reviews and a \\$30 game with 1,000
  reviews earned very different amounts.
- Only store-page information is used. Wishlists, trailers, marketing, streamers and the game itself are invisible.
- Store pages change after launch (more languages, screenshots, achievements), so some signals are slightly optimistic.
- Studio "best previous game" reviews are counted as of the data snapshot, not at launch time.
- Patterns are correlations. Adding Linux support will not cause success on its own.
""")
