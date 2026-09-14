"""Train, evaluate and export the final model.

  0. train 2018-2022, predict 2023: choose the blend weight
  1. train 2018-2023, predict 2024: held-out test metrics
  2. train 2018-2024: model saved to models/gamesense.joblib

Usage: python -m src.train
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURE_COLS, TEXT_COL, TextScorer, build_base_features, game_text
from src.metrics import tier_metrics

QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}


def load_params():
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    return best["xgboost"]["params"], best["lightgbm"]["params"]


def make_xgb(p):
    return xgb.XGBClassifier(**p, objective="multi:softprob", tree_method="hist", n_jobs=-1,
                             random_state=config.RANDOM_STATE)


def make_lgb(p):
    return lgb.LGBMClassifier(**p, objective="multiclass", subsample_freq=1, verbose=-1, n_jobs=-1,
                              random_state=config.RANDOM_STATE)


def make_quantile(alpha):
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=700, learning_rate=0.04,
                             num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=config.RANDOM_STATE)


def featurize(train_df, other_df=None):
    """Features for the training frame (out-of-fold text score) and optionally a later frame."""
    scorer = TextScorer()
    Xtr = build_base_features(train_df)
    Xtr[TEXT_COL] = scorer.fit_oof(game_text(train_df), np.log1p(train_df.total_reviews.values))
    Xtr = Xtr[FEATURE_COLS]
    if other_df is None:
        return scorer, Xtr, None
    Xo = build_base_features(other_df)
    Xo[TEXT_COL] = scorer.predict(game_text(other_df))
    return scorer, Xtr, Xo[FEATURE_COLS]


def fit_ensemble(X, y, xgb_p, lgb_p):
    return make_xgb(xgb_p).fit(X, y), make_lgb(lgb_p).fit(X, y)


def blend(models, X, w_xgb):
    return w_xgb * models[0].predict_proba(X) + (1 - w_xgb) * models[1].predict_proba(X)


def main():
    t0 = time.time()
    df = pd.read_parquet(config.PROCESSED_PATH)
    xgb_p, lgb_p = load_params()
    yrs = df.year
    train = df[yrs.between(*config.TRAIN_YEARS)].reset_index(drop=True)
    valid = df[yrs.between(*config.VALID_YEARS)].reset_index(drop=True)
    trval = df[yrs.between(config.TRAIN_YEARS[0], config.VALID_YEARS[1])].reset_index(drop=True)
    test = df[yrs.between(*config.TEST_YEARS)].reset_index(drop=True)

    # Stage 0
    print("stage 0: validation (train 2018-22 -> 2023)")
    _, Xtr, Xva = featurize(train, valid)
    ens = fit_ensemble(Xtr, train.tier.values, xgb_p, lgb_p)
    p_x, p_l = ens[0].predict_proba(Xva), ens[1].predict_proba(Xva)
    w_grid = np.linspace(0, 1, 11)
    losses = [log_loss(valid.tier, w * p_x + (1 - w) * p_l) for w in w_grid]
    w_xgb = float(w_grid[int(np.argmin(losses))])
    p_va = w_xgb * p_x + (1 - w_xgb) * p_l
    print(f"  blend w_xgb={w_xgb:.1f}  valid log loss={min(losses):.4f}  "
          f"macro F1={f1_score(valid.tier, p_va.argmax(1), average='macro'):.3f}")

    # Stage 1
    print("stage 1: test (train 2018-23 -> 2024)")
    _, Xtv, Xte = featurize(trval, test)
    ytv, yte = trval.tier.values, test.tier.values
    rev_te = test.total_reviews.values
    base_cols = [c for c in FEATURE_COLS if c != TEXT_COL]

    comparison = {}
    baselines = {
        "Baseline: majority class": (DummyClassifier(strategy="prior"), base_cols),
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)), FEATURE_COLS),
        "Random Forest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1,
                                                 random_state=config.RANDOM_STATE), FEATURE_COLS),
    }
    for name, (m, cols) in baselines.items():
        m.fit(Xtv[cols], ytv)
        comparison[name] = tier_metrics(yte, m.predict_proba(Xte[cols]), reviews_true=rev_te)

    ens = fit_ensemble(Xtv, ytv, xgb_p, lgb_p)
    p_x, p_l = ens[0].predict_proba(Xte), ens[1].predict_proba(Xte)
    comparison["XGBoost (tuned)"] = tier_metrics(yte, p_x, reviews_true=rev_te)
    comparison["LightGBM (tuned)"] = tier_metrics(yte, p_l, reviews_true=rev_te)
    p_te = w_xgb * p_x + (1 - w_xgb) * p_l
    comparison["GameSense ensemble"] = tier_metrics(yte, p_te, reviews_true=rev_te)
    final_pred = p_te.argmax(1)

    comp = pd.DataFrame(comparison).T.round(4)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(config.REPORTS_DIR / "model_comparison_test2024.csv")
    print(comp[["accuracy", "macro_f1", "qwk", "within_1_tier", "auc_ovr", "auc_success_plus", "log_loss"]].to_string())

    # Review-count range via quantile regression
    y_log_tv = np.log1p(trval.total_reviews.values)
    q = {k: make_quantile(a).fit(Xtv, y_log_tv).predict(Xte) for k, a in QUANTILES.items()}
    y_log_te = np.log1p(rev_te)
    reg_metrics = {
        "median_abs_error_log": float(np.median(np.abs(q["mid"] - y_log_te))),
        "within_x3_of_actual": float(np.mean(np.abs(q["mid"] - y_log_te) <= np.log(3))),
        "interval_80_coverage": float(np.mean((y_log_te >= q["low"]) & (y_log_te <= q["high"]))),
        "spearman": float(pd.Series(q["mid"]).corr(pd.Series(y_log_te), method="spearman")),
    }
    print("review-range model:", {k: round(v, 3) for k, v in reg_metrics.items()})

    test_out = test[["app_id", "name", "total_reviews", "tier"]].copy()
    for i, n in enumerate(config.TIER_NAMES):
        test_out[f"p_{n.lower()}"] = p_te[:, i]
    test_out["pred_tier"] = final_pred
    test_out["pred_reviews_mid"] = np.expm1(q["mid"])
    test_out.to_parquet(config.REPORTS_DIR / "test2024_predictions.parquet", index=False)

    # Stage 2
    print("stage 2: refit on 2018-2024")
    scorer, Xall, _ = featurize(df)
    yall = df.tier.values
    ens = fit_ensemble(Xall, yall, xgb_p, lgb_p)
    y_log_all = np.log1p(df.total_reviews.values)
    quantile_models = {k: make_quantile(a).fit(Xall, y_log_all) for k, a in QUANTILES.items()}

    bundle = {
        "text_scorer": scorer,
        "xgb": ens[0],
        "lgb": ens[1],
        "w_xgb": w_xgb,
        "quantile_models": quantile_models,
        "feature_cols": FEATURE_COLS,
        "tier_names": config.TIER_NAMES,
        "train_rows": len(df),
        "prediction_year": config.YEAR_MAX,
        "feature_medians": Xall.median().to_dict(),
    }
    joblib.dump(bundle, config.MODELS_DIR / "gamesense.joblib", compress=3)

    summary = {
        "rows_total": len(df), "rows_trainval": len(trval), "rows_test": len(test),
        "blend_w_xgb": w_xgb,
        "test_2024": comp.to_dict(orient="index"),
        "review_range_test_2024": reg_metrics,
        "confusion_matrix_test_2024": confusion_matrix(yte, final_pred).tolist(),
        "test_tier_distribution": pd.Series(yte).value_counts(normalize=True).sort_index().round(4).tolist(),
    }
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
