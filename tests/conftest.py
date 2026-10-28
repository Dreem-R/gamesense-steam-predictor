import pytest

from gamesense.game import Game
from gamesense.predictor import GameSensePredictor
from gamesense.presets import PRESETS


@pytest.fixture(scope="session")
def predictor() -> GameSensePredictor:
    return GameSensePredictor()


@pytest.fixture(params=list(PRESETS), ids=list(PRESETS))
def preset_game(request: pytest.FixtureRequest) -> Game:
    return PRESETS[request.param]
