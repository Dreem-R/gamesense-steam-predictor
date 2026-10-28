"""HTTP API for GameSense predictions.

Run locally with: uvicorn gamesense.api.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.responses import RedirectResponse

from gamesense import config, datasets
from gamesense.api import schemas
from gamesense.features import FEATURE_COLS, slug
from gamesense.game import Game
from gamesense.predictor import GameSensePredictor
from gamesense.presets import PRESETS
from gamesense.reliability import (
    MIN_CALIBRATION_GAMES,
    calibration_check,
    confidence_level,
    input_warnings,
    tier_track_record,
)

TEST_YEAR = config.TEST_YEARS[0]
REPORTED_METRICS = ["accuracy", "macro_f1", "qwk", "within_1_tier", "auc_success_plus"]


@dataclass(frozen=True)
class Resources:
    predictor: GameSensePredictor
    market: pd.DataFrame
    test_predictions: pd.DataFrame
    metrics: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.resources = Resources(
        predictor=GameSensePredictor(),
        market=datasets.load_market(),
        test_predictions=datasets.load_test_predictions(),
        metrics=datasets.load_metrics(),
    )
    yield


app = FastAPI(
    title="GameSense API",
    version=version("gamesense"),
    description="Predicts how a game will perform on Steam from information available before launch.",
    lifespan=lifespan,
)


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


Resource = Annotated[Resources, Depends(get_resources)]
GameBody = Annotated[
    Game,
    Body(
        openapi_examples={
            slug(name): {"summary": name, "value": game.model_dump()}
            for name, game in PRESETS.items()
            if game != Game()
        }
    ),
]


@app.get("/", include_in_schema=False)
def docs_redirect() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> schemas.Health:
    return schemas.Health()


@app.get("/model")
def model_info(r: Resource) -> schemas.ModelInfo:
    m = r.predictor.model
    final = r.metrics["test_2024"]["GameSense ensemble"]
    return schemas.ModelInfo(
        version=app.version,
        train_rows=m.train_rows,
        prediction_year=m.prediction_year,
        tiers=m.tier_names,
        features=len(FEATURE_COLS),
        test_year=TEST_YEAR,
        test_metrics={k: final[k] for k in REPORTED_METRICS},
    )


@app.post("/predict")
def predict(game: GameBody, r: Resource) -> schemas.Prediction:
    pred = r.predictor.predict(game)
    level, detail = confidence_level(pred.proba)
    calib = calibration_check(r.test_predictions, pred.success_chance)
    return schemas.Prediction(
        tier=pred.tier_name,
        probabilities=dict(zip(config.TIER_NAMES, pred.proba.tolist(), strict=True)),
        success_chance=pred.success_chance,
        reviews=schemas.ReviewRange(low=pred.reviews_low, expected=pred.reviews_mid, high=pred.reviews_high),
        used_description=pred.used_text,
        reliability=schemas.Reliability(
            confidence=level,
            confidence_detail=detail,
            tier_accuracy=tier_track_record(r.test_predictions, pred.tier)["exact"],
            similar_predictions_rate=calib["actual_rate"] if calib["games"] >= MIN_CALIBRATION_GAMES else None,
        ),
        warnings=input_warnings(game, r.market),
    )


@app.post("/explain")
def explain(game: GameBody, r: Resource, top_k: Annotated[int, Query(ge=1, le=30)] = 8) -> list[schemas.Contribution]:
    return [schemas.Contribution.of(c) for c in r.predictor.explain(game, top_k=top_k)]


@app.post("/what-if")
def what_if(game: GameBody, r: Resource) -> schemas.WhatIf:
    scored, plan = r.predictor.what_if(game)
    combined = None
    if plan:
        combined = schemas.CombinedPlan(
            change_ids=plan.change_ids,
            success_after=plan.success_after,
            reviews_after=plan.reviews_after,
            tier_after=plan.tier_after,
        )
    return schemas.WhatIf(changes=[schemas.ScoredChange.of(s) for s in scored], combined=combined)
