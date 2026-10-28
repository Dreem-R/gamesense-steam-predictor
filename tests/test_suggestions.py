import pytest

from gamesense.game import Game
from gamesense.suggestions import Change, ScoredChange, candidate_changes


def changes_for(**fields) -> dict[str, Change]:
    return {c.id: c for c in candidate_changes(Game(**fields))}


def test_cheap_games_get_a_price_rise():
    assert changes_for(price=2.99)["raise_price"].updates == {"price": 14.99}


def test_expensive_games_get_a_price_cut():
    assert changes_for(price=39.99)["lower_price"].updates == {"price": 24.99}


def test_free_games_get_no_price_change():
    assert not {"raise_price", "lower_price"} & changes_for(price=0).keys()


def test_existing_features_are_not_suggested_again():
    complete = changes_for(
        n_screenshots=12,
        has_website=True,
        achievements=30,
        n_languages=10,
        linux=True,
        mac=True,
        price=19.99,
        self_published=False,
        categories=["Single-player", "Steam Cloud", "Full controller support", "Co-op"],
    )
    assert complete == {}


def test_publisher_change_is_high_effort_and_only_for_self_published():
    assert changes_for(self_published=True)["publisher"].effort == "High"
    assert "publisher" not in changes_for(self_published=False)


def test_applying_a_change_keeps_existing_categories():
    game = Game(categories=["Single-player", "Steam Workshop"])
    after = changes_for(categories=game.categories)["steam_cloud"].apply(game)
    assert set(after.categories) == {"Single-player", "Steam Workshop", "Steam Cloud"}


def test_applying_the_publisher_change_switches_history():
    game = Game(self_published=True, dev_prior_games=1)
    after = changes_for(self_published=True, dev_prior_games=1)["publisher"].apply(game)
    assert not after.self_published
    assert after.pub_prior_games == 50


def scored(success_after: float, reviews_after: float) -> ScoredChange:
    return ScoredChange(Change("website", "Store page", "Low"), 0.20, success_after, 100.0, reviews_after)


@pytest.mark.parametrize(
    ("success_after", "reviews_after", "helps"),
    [
        (0.215, 100, True),  # +1.5 points
        (0.205, 100, False),  # too small
        (0.198, 120, True),  # +19% reviews, chance about the same
        (0.19, 200, False),  # more reviews but chance falls a full point
        (0.20, 110, False),  # +9% reviews only
    ],
)
def test_helps_threshold(success_after, reviews_after, helps):
    assert scored(success_after, reviews_after).helps is helps
