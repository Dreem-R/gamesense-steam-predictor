"""Baseline model comparison: train on 2018-2022, validate on 2023.

Usage: python -m src.experiments
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import BASE_COLS, CATEGORY_COLS, GENRE_COLS, TEXT_COL, TextScorer, build_base_features, game_text
from src.metrics import tier_metrics


def load_split():
    df = pd.read_parquet(config.PROCESSED_PATH)
    tr = df[df.year.between(*config.TRAIN_YEARS)].reset_index(drop=True)
    va = df[df.year.between(*config.VALID_YEARS)].reset_index(drop=True)
    return tr, va


def main():
    tr, va = load_split()
    print(f"train {len(tr):,} | valid {len(va):,}")
    Xtr, Xva = build_base_features(tr), build_base_features(va)
    ytr, yva = tr.tier.values, va.tier.values

    t = time.time()
    scorer = TextScorer()
    Xtr[TEXT_COL] = scorer.fit_oof(game_text(tr), np.log1p(tr.total_reviews.values))
    Xva[TEXT_COL] = scorer.predict(game_text(va))
    print(f"text scorer: {time.time() - t:.0f}s")
    pos, neg = scorer.top_terms(15)
    print("words linked to MORE reviews:", [w for w, _ in pos])
    print("words linked to FEWER reviews:", [w for w, _ in neg])

    no_text = BASE_COLS + GENRE_COLS + CATEGORY_COLS
    with_text = no_text + [TEXT_COL]

    models = {
        "Baseline (majority class)": (DummyClassifier(strategy="prior"), no_text),
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)), with_text),
        "Random Forest": (RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                                 random_state=config.RANDOM_STATE), with_text),
        "XGBoost": (xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                      random_state=config.RANDOM_STATE), with_text),
        "LightGBM (no text)": (lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                  verbose=-1, random_state=config.RANDOM_STATE), no_text),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        verbose=-1, random_state=config.RANDOM_STATE), with_text),
    }
    rows = []
    for name, (model, cols) in models.items():
        t = time.time()
        model.fit(Xtr[cols], ytr)
        proba = model.predict_proba(Xva[cols])
        m = tier_metrics(yva, proba, reviews_true=va.total_reviews.values)
        m["model"], m["seconds"] = name, round(time.time() - t, 1)
        rows.append(m)
        print(f"{name:28s} macroF1={m['macro_f1']:.3f} qwk={m['qwk']:.3f} auc={m['auc_ovr']:.3f} "
              f"logloss={m['log_loss']:.3f} spearman={m['spearman_reviews']:.3f}")

    res = pd.DataFrame(rows).set_index("model").round(4)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.REPORTS_DIR / "model_comparison_validation.csv")
    print(res.to_string())


if __name__ == "__main__":
    main()
