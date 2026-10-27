from dataclasses import replace

import joblib
import pytest

from gamesense import config
from gamesense.bundle import IncompatibleModelError, ModelBundle
from gamesense.features import FEATURE_COLS


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return ModelBundle.load()


def test_shipped_model_matches_the_code(bundle):
    assert bundle.feature_cols == FEATURE_COLS
    assert bundle.tier_names == config.TIER_NAMES
    assert set(bundle.quantile_models) == {"low", "mid", "high"}


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_cols": FEATURE_COLS[:-1]},
        {"tier_names": ["Low", "High"]},
        {"quantile_models": {}},
        {"w_xgb": 1.5},
    ],
)
def test_mismatched_model_is_rejected(bundle, changes):
    with pytest.raises(IncompatibleModelError, match="Retrain"):
        replace(bundle, **changes).check()


def test_file_with_missing_parts_is_rejected(tmp_path):
    path = tmp_path / "partial.joblib"
    joblib.dump({"w_xgb": 0.5, "train_rows": 10}, path)
    with pytest.raises(IncompatibleModelError, match="missing"):
        ModelBundle.load(path)
