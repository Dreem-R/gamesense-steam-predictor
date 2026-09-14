"""Optuna search for XGBoost and LightGBM (train 2018-2022, validate 2023).

Writes models/best_params.json.

Usage: python -m src.tune --trials 40
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss

from src import config
from src.experiments import load_split
from src.features import FEATURE_COLS, TEXT_COL, TextScorer, build_base_features, game_text

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare():
    tr, va = load_split()
    Xtr, Xva = build_base_features(tr), build_base_features(va)
    scorer = TextScorer()
    Xtr[TEXT_COL] = scorer.fit_oof(game_text(tr), np.log1p(tr.total_reviews.values))
    Xva[TEXT_COL] = scorer.predict(game_text(va))
    return Xtr[FEATURE_COLS], tr.tier.values, Xva[FEATURE_COLS], va.tier.values


def xgb_objective(Xtr, ytr, Xva, yva):
    def objective(trial):
        params = dict(
            n_estimators=3000, early_stopping_rounds=100, tree_method="hist", n_jobs=-1,
            objective="multi:softprob", eval_metric="mlogloss", random_state=config.RANDOM_STATE,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            gamma=trial.suggest_float("gamma", 0, 5),
        )
        m = xgb.XGBClassifier(**params).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        trial.set_user_attr("n_estimators", int(m.best_iteration) + 1)
        return log_loss(yva, m.predict_proba(Xva))
    return objective


def lgb_objective(Xtr, ytr, Xva, yva):
    def objective(trial):
        params = dict(
            n_estimators=3000, objective="multiclass", verbose=-1, n_jobs=-1,
            random_state=config.RANDOM_STATE, subsample_freq=1,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
        m = lgb.LGBMClassifier(**params).fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("n_estimators", int(m.best_iteration_))
        return log_loss(yva, m.predict_proba(Xva))
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    data = prepare()
    best = {}
    for name, objective in [("xgboost", xgb_objective), ("lightgbm", lgb_objective)]:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(objective(*data), n_trials=args.trials, show_progress_bar=False)
        params = dict(study.best_params, n_estimators=study.best_trial.user_attrs["n_estimators"])
        best[name] = {"params": params, "valid_log_loss": study.best_value}
        print(f"{name}: best valid log loss {study.best_value:.4f}  {params}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "best_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
