"""Changes a developer could make to a game, and how the model scored them."""

from dataclasses import dataclass, field
from typing import Any, Literal

from gamesense.game import Game

ChangeId = Literal[
    "more_screenshots",
    "website",
    "steam_cloud",
    "achievements",
    "controller",
    "localisation",
    "linux",
    "macos",
    "raise_price",
    "lower_price",
    "online_coop",
    "publisher",
]
Area = Literal["Store page", "Steam features", "Reach", "Pricing", "Design", "Business"]
Effort = Literal["Low", "Medium", "High"]

# A change helps if it adds at least 1 point of success chance, or 15% more expected
# reviews without lowering that chance, so games with very low or very high chances
# still get useful comparisons.
MIN_UPLIFT = 0.01
MIN_REVIEW_RATIO = 1.15


@dataclass(frozen=True)
class Change:
    id: ChangeId
    area: Area
    effort: Effort
    updates: dict[str, Any] = field(default_factory=dict)

    def apply(self, game: Game) -> Game:
        updates = dict(self.updates)
        if "categories" in updates:
            updates["categories"] = sorted(set(game.categories) | set(updates["categories"]))
        return game.with_changes(**updates)


@dataclass(frozen=True)
class ScoredChange:
    change: Change
    success_before: float
    success_after: float
    reviews_before: float
    reviews_after: float

    @property
    def uplift(self) -> float:
        return self.success_after - self.success_before

    @property
    def reviews_ratio(self) -> float:
        return (self.reviews_after + 1) / (self.reviews_before + 1)

    @property
    def helps(self) -> bool:
        return self.uplift >= MIN_UPLIFT or (self.reviews_ratio >= MIN_REVIEW_RATIO and self.uplift > -0.005)


@dataclass(frozen=True)
class CombinedPlan:
    """Every low and medium effort change that helped, applied together."""

    change_ids: list[ChangeId]
    success_after: float
    reviews_after: float
    tier_after: str


def candidate_changes(game: Game) -> list[Change]:
    cats = set(game.categories)
    options = []
    if game.n_screenshots < 10:
        options.append(Change("more_screenshots", "Store page", "Low", {"n_screenshots": 10}))
    if not game.has_website:
        options.append(Change("website", "Store page", "Low", {"has_website": True}))
    if "Steam Cloud" not in cats:
        options.append(Change("steam_cloud", "Steam features", "Low", {"categories": ["Steam Cloud"]}))
    if game.achievements == 0:
        options.append(
            Change(
                "achievements", "Steam features", "Medium", {"achievements": 25, "categories": ["Steam Achievements"]}
            )
        )
    if not cats & {"Full controller support", "Partial Controller Support"}:
        options.append(Change("controller", "Steam features", "Medium", {"categories": ["Full controller support"]}))
    if game.n_languages < 8:
        options.append(Change("localisation", "Reach", "Medium", {"n_languages": 8}))
    if not game.linux:
        options.append(Change("linux", "Reach", "Medium", {"linux": True}))
    if not game.mac:
        options.append(Change("macos", "Reach", "Medium", {"mac": True}))
    if 0 < game.price < 10:
        options.append(Change("raise_price", "Pricing", "Low", {"price": 14.99}))
    if game.price > 30:
        options.append(Change("lower_price", "Pricing", "Low", {"price": 24.99}))
    if not cats & {"Co-op", "Online Co-op"}:
        options.append(
            Change("online_coop", "Design", "High", {"categories": ["Multi-player", "Co-op", "Online Co-op"]})
        )
    if game.self_published:
        options.append(
            Change(
                "publisher",
                "Business",
                "High",
                {"self_published": False, "pub_prior_games": 50, "pub_prior_best_reviews": 5000},
            )
        )
    return options
