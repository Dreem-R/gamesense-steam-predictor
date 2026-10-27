"""The trained model file: loading, saving and checking it against the current feature code."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier

from gamesense import config
from gamesense.features import FEATURE_COLS, TextScorer

QUANTILE_KEYS = {"low", "mid", "high"}


class IncompatibleModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    text_scorer: TextScorer
    xgb: XGBClassifier
    lgb: LGBMClassifier
    w_xgb: float
    quantile_models: dict[str, LGBMRegressor]
    feature_cols: list[str]
    tier_names: list[str]
    train_rows: int
    prediction_year: int
    feature_medians: dict[str, float]

    @classmethod
    def load(cls, path: Path = config.MODEL_PATH) -> Self:
        # Stored as a plain dict so the file does not depend on this class's import path.
        raw = joblib.load(path)
        missing = sorted({f.name for f in fields(cls)} - raw.keys())
        if missing:
            raise IncompatibleModelError(f"{path} is missing {', '.join(missing)}")
        bundle = cls(**{f.name: raw[f.name] for f in fields(cls)})
        bundle.check()
        return bundle

    def save(self, path: Path = config.MODEL_PATH) -> None:
        self.check()
        joblib.dump({f.name: getattr(self, f.name) for f in fields(self)}, path, compress=3)

    def check(self) -> None:
        problems = []
        if self.feature_cols != FEATURE_COLS:
            problems.append("its features differ from gamesense.features.FEATURE_COLS")
        if self.tier_names != config.TIER_NAMES:
            problems.append("its tiers differ from gamesense.config.TIER_NAMES")
        if set(self.quantile_models) != QUANTILE_KEYS:
            problems.append(f"it needs quantile models {sorted(QUANTILE_KEYS)}")
        if not 0 <= self.w_xgb <= 1:
            problems.append("its blend weight is outside 0-1")
        if problems:
            raise IncompatibleModelError(
                "The model does not match this code: " + "; ".join(problems) + ". "
                "Retrain with python -m gamesense.pipeline.train."
            )
