import lightgbm  # noqa: F401, I001

import pytest

from gamesense.predictor import GameSensePredictor, default_game
from gamesense.presets import PRESETS


@pytest.fixture(scope="session")
def predictor() -> GameSensePredictor:
    return GameSensePredictor()


@pytest.fixture(params=list(PRESETS), ids=list(PRESETS))
def preset_game(request: pytest.FixtureRequest) -> dict:
    return default_game(**PRESETS[request.param])
