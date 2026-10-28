"""A game as described before launch: the input to every prediction."""

from collections.abc import Iterable
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.config import JsonDict

from gamesense import config


def _allowed(values: list[str]) -> JsonDict:
    """Lists the allowed values in the API schema. The validators on Game enforce them."""
    enum: list[JsonValue] = list(values)
    return {"items": {"type": "string", "enum": enum}}


class Game(BaseModel):
    """A game as described before launch. Every field has a default, so a request only needs the fields it
    wants to set."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("My Game", description="Echoed in the app only. Not used by the model.")
    short_description: str = Field(
        "", description="The one-line summary on the store page. Read by the text model together with `about`."
    )
    about: str = Field(
        "",
        description="The full store page description. The text model is only used when `short_description` and "
        "`about` together have at least 20 words.",
    )
    genres: list[str] = Field(
        default_factory=lambda: ["Indie"],
        min_length=1,
        description="Genres as listed on Steam. At least one is required.",
        json_schema_extra=_allowed(config.GENRES),
    )
    categories: list[str] = Field(
        default_factory=lambda: ["Single-player"],
        description="Steam features such as multiplayer modes, achievements, cloud saves and controller support.",
        json_schema_extra=_allowed(config.CATEGORIES),
    )
    price: float = Field(9.99, ge=0, le=100, description="Launch price in USD. 0 means free to play.")
    month: int = Field(10, ge=1, le=12, description="Release month, 1 to 12.")
    required_age: int = Field(0, ge=0, le=21, description="Minimum age on the store page. 17 or more counts as mature.")
    windows: bool = Field(True, description="Available on Windows.")
    mac: bool = Field(False, description="Available on macOS.")
    linux: bool = Field(False, description="Available on Linux, which includes Steam Deck support.")
    achievements: int = Field(0, ge=0, description="Number of Steam Achievements.")
    n_languages: int = Field(1, ge=0, description="Languages with interface or subtitle support.")
    n_audio_languages: int = Field(0, ge=0, description="Languages with full voice-over.")
    n_screenshots: int = Field(6, ge=0, description="Screenshots on the store page.")
    has_website: bool = Field(False, description="The game has an official website.")
    dev_prior_games: int = Field(0, ge=0, description="Games the studio released on Steam before this one.")
    dev_prior_best_reviews: int = Field(
        0, ge=0, description="Reviews on the studio's best previous game. Ignored when `dev_prior_games` is 0."
    )
    self_published: bool = Field(
        True, description="The studio publishes the game itself, so the publisher history is the studio's."
    )
    pub_prior_games: int = Field(
        0, ge=0, description="Games the publisher released before this one. Ignored when `self_published` is true."
    )
    pub_prior_best_reviews: int = Field(
        0,
        ge=0,
        description="Reviews on the publisher's best previous game. Ignored when `self_published` is true.",
    )

    @field_validator("genres")
    @classmethod
    def known_genres(cls, genres: list[str]) -> list[str]:
        return _known(genres, config.GENRES, "genres")

    @field_validator("categories")
    @classmethod
    def known_categories(cls, categories: list[str]) -> list[str]:
        return _known(categories, config.CATEGORIES, "categories")

    @model_validator(mode="after")
    def consistent_history(self) -> Self:
        if self.dev_prior_games == 0:
            self.dev_prior_best_reviews = 0
        # A self-published game's publisher is the studio, so it shares the studio's history.
        if self.self_published:
            self.pub_prior_games = self.dev_prior_games
            self.pub_prior_best_reviews = self.dev_prior_best_reviews
        return self

    @property
    def about_words(self) -> int:
        return len(self.about.split())

    def with_changes(self, **changes: Any) -> Self:
        """A validated copy with some fields replaced."""
        return self.model_validate(self.model_dump() | changes)

    def to_row(self) -> dict[str, Any]:
        """The game in the schema of the cleaned training data."""
        return self.model_dump() | {"about_words": self.about_words, "year": config.YEAR_MAX}


def to_frame(games: Iterable[Game]) -> pd.DataFrame:
    return pd.DataFrame([g.to_row() for g in games])


def _known(values: list[str], vocabulary: list[str], field: str) -> list[str]:
    unknown = sorted(set(values) - set(vocabulary))
    if unknown:
        raise ValueError(f"unknown {field}: {', '.join(unknown)}")
    return list(dict.fromkeys(values))
