"""A game as described before launch: the input to every prediction."""

from collections.abc import Iterable
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gamesense import config


class Game(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "My Game"
    short_description: str = ""
    about: str = Field("", description="Full store page description.")
    genres: list[str] = Field(default_factory=lambda: ["Indie"], min_length=1)
    categories: list[str] = Field(default_factory=lambda: ["Single-player"], description="Steam features.")
    price: float = Field(9.99, ge=0, le=100, description="USD. 0 means free to play.")
    month: int = Field(10, ge=1, le=12, description="Release month.")
    required_age: int = Field(0, ge=0, le=21)
    windows: bool = True
    mac: bool = False
    linux: bool = False
    achievements: int = Field(0, ge=0)
    n_languages: int = Field(1, ge=0)
    n_audio_languages: int = Field(0, ge=0)
    n_screenshots: int = Field(6, ge=0)
    has_website: bool = False
    dev_prior_games: int = Field(0, ge=0, description="Games the studio released on Steam before this one.")
    dev_prior_best_reviews: int = Field(0, ge=0, description="Reviews on the studio's best previous game.")
    self_published: bool = True
    pub_prior_games: int = Field(0, ge=0)
    pub_prior_best_reviews: int = Field(0, ge=0)

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
