"""Response models for the HTTP API. Requests use gamesense.game.Game directly."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from gamesense import predictor, suggestions
from gamesense.reliability import Confidence


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfo(BaseModel):
    version: str
    train_rows: int
    prediction_year: int
    tiers: list[str]
    features: int
    test_year: int
    test_metrics: dict[str, float] = Field(description="Scores of the final model on the held-out test year.")


class ReviewRange(BaseModel):
    low: float = Field(description="10th percentile estimate.")
    expected: float = Field(description="Median estimate.")
    high: float = Field(description="90th percentile estimate.")


class Reliability(BaseModel):
    confidence: Confidence
    confidence_detail: str
    tier_accuracy: float = Field(description="Share of held-out games placed in this tier that ended up in it.")
    similar_predictions_rate: float | None = Field(
        description="Share of held-out games with a similar predicted chance that reached 100+ reviews. "
        "Null when too few held-out games had a similar chance."
    )


class Prediction(BaseModel):
    tier: str
    probabilities: dict[str, float]
    success_chance: float = Field(description="Probability of reaching 100+ reviews.")
    reviews: ReviewRange
    used_description: bool = Field(description="False when the description is too short for the text model.")
    reliability: Reliability
    warnings: list[str] = Field(description="Inputs that are rare in the training data.")


class Contribution(BaseModel):
    feature: str
    label: str
    value: str
    effect: float = Field(description="Change in log expected reviews compared with an average game.")
    multiplier: float = Field(description="The same effect as a multiplier on expected reviews.")

    @classmethod
    def of(cls, c: predictor.Contribution) -> "Contribution":
        return cls(feature=c.feature, label=c.label, value=c.value, effect=c.effect, multiplier=c.multiplier)


class ScoredChange(BaseModel):
    id: str
    area: str
    effort: suggestions.Effort
    updates: dict[str, Any] = Field(
        description="Fields changed on the game. Categories are added to the existing ones."
    )
    success_before: float
    success_after: float
    uplift: float
    reviews_before: float
    reviews_after: float
    helps: bool

    @classmethod
    def of(cls, s: suggestions.ScoredChange) -> "ScoredChange":
        return cls(
            id=s.change.id,
            area=s.change.area,
            effort=s.change.effort,
            updates=s.change.updates,
            success_before=s.success_before,
            success_after=s.success_after,
            uplift=s.uplift,
            reviews_before=s.reviews_before,
            reviews_after=s.reviews_after,
            helps=s.helps,
        )


class CombinedPlan(BaseModel):
    change_ids: list[str]
    success_after: float
    reviews_after: float
    tier_after: str


class WhatIf(BaseModel):
    changes: list[ScoredChange] = Field(description="Each change scored on its own, best first.")
    combined: CombinedPlan | None = Field(
        description="Every helpful low and medium effort change applied together. Null when fewer than two helped."
    )
