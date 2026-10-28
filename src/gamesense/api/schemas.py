"""Response models for the HTTP API. Requests use gamesense.game.Game directly."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from gamesense import config, predictor, suggestions
from gamesense.reliability import Confidence

TIERS = ", ".join(config.TIER_NAMES)


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfo(BaseModel):
    """The model behind the API and how it scored on releases it did not see during training."""

    version: str = Field(description="Version of the gamesense package.", examples=["0.1.0"])
    train_rows: int = Field(description="Games the final model was trained on.", examples=[69127])
    prediction_year: int = Field(description="Release year the model assumes for every game.", examples=[2024])
    tiers: list[str] = Field(description="Outcome tiers, lowest first.", examples=[config.TIER_NAMES])
    features: int = Field(description="Number of model features.", examples=[65])
    test_year: int = Field(description="Release year held out for evaluation.", examples=[2024])
    test_metrics: dict[str, float] = Field(
        description="Accuracy, macro F1, quadratic weighted kappa, share within one tier, and ROC-AUC for "
        "reaching 100+ reviews.",
        examples=[
            {"accuracy": 0.666, "macro_f1": 0.577, "qwk": 0.663, "within_1_tier": 0.968, "auc_success_plus": 0.9}
        ],
    )


class ReviewRange(BaseModel):
    low: float = Field(description="10th percentile estimate.", examples=[0.1])
    expected: float = Field(description="Median estimate.", examples=[9.3])
    high: float = Field(description="90th percentile estimate.", examples=[72.4])


class Reliability(BaseModel):
    """How far to trust a prediction, based on held-out results."""

    confidence: Confidence
    confidence_detail: str = Field(examples=["Split between Flop (50%) and Niche (31%)."])
    tier_accuracy: float = Field(
        description="Share of held-out games placed in this tier that ended up in it.", examples=[0.79]
    )
    similar_predictions_rate: float | None = Field(
        description="Share of held-out games with a similar predicted chance that reached 100+ reviews. "
        "Null when too few held-out games had a similar chance.",
        examples=[0.13],
    )


class Prediction(BaseModel):
    """The predicted outcome for one game."""

    tier: str = Field(description=f"Most likely outcome tier: one of {TIERS}.", examples=["Flop"])
    probabilities: dict[str, float] = Field(
        description="Probability of each tier. They sum to 1.",
        examples=[{"Flop": 0.502, "Niche": 0.31, "Success": 0.174, "Hit": 0.015}],
    )
    success_chance: float = Field(description="Probability of reaching 100+ reviews.", examples=[0.188])
    reviews: ReviewRange
    used_description: bool = Field(description="False when the description is too short for the text model.")
    reliability: Reliability
    warnings: list[str] = Field(
        description="Inputs that are rare in the training data, where predictions are less reliable.",
        examples=[["Very few games in the data are priced above $60."]],
    )


class Contribution(BaseModel):
    """One feature's effect on expected reviews."""

    feature: str = Field(description="Model feature name.", examples=["n_screenshots"])
    label: str = Field(description="Readable name.", examples=["Screenshots on store page"])
    value: str = Field(description="This game's value for the feature.", examples=["8"])
    effect: float = Field(description="Change in log expected reviews compared with an average game.", examples=[-0.21])
    multiplier: float = Field(description="The same effect as a multiplier on expected reviews.", examples=[0.81])

    @classmethod
    def of(cls, c: predictor.Contribution) -> "Contribution":
        return cls(feature=c.feature, label=c.label, value=c.value, effect=c.effect, multiplier=c.multiplier)


class ScoredChange(BaseModel):
    """One change applied to the game on its own, and how the model scored it."""

    id: suggestions.ChangeId
    area: suggestions.Area
    effort: suggestions.Effort
    updates: dict[str, Any] = Field(
        description="Fields set by the change. Categories are added to the game's existing ones.",
        examples=[{"n_screenshots": 10}],
    )
    success_before: float = Field(description="Chance of 100+ reviews without the change.", examples=[0.188])
    success_after: float = Field(description="Chance of 100+ reviews with the change.", examples=[0.214])
    uplift: float = Field(description="success_after minus success_before.", examples=[0.026])
    reviews_before: float = Field(description="Expected reviews without the change.", examples=[9.3])
    reviews_after: float = Field(description="Expected reviews with the change.", examples=[11.2])
    helps: bool = Field(description="Whether the change passes the bar described above.")

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
    """Every helpful low and medium effort change applied together."""

    change_ids: list[suggestions.ChangeId]
    success_after: float = Field(description="Chance of 100+ reviews with all the changes.", examples=[0.331])
    reviews_after: float = Field(description="Expected reviews with all the changes.", examples=[30.4])
    tier_after: str = Field(description=f"Most likely tier with all the changes: one of {TIERS}.", examples=["Niche"])


class WhatIf(BaseModel):
    changes: list[ScoredChange] = Field(description="Each change scored on its own, best first.")
    combined: CombinedPlan | None = Field(
        description="All helpful low and medium effort changes applied together. Null when fewer than two helped."
    )
