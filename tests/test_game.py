import pytest
from pydantic import ValidationError

from gamesense import config
from gamesense.game import Game, to_frame


def test_every_field_is_documented():
    undocumented = [name for name, f in Game.model_fields.items() if not f.description]
    assert undocumented == []


def test_defaults_are_valid():
    game = Game()
    assert game.genres == ["Indie"]
    assert game.categories == ["Single-player"]


@pytest.mark.parametrize(
    "bad",
    [
        {"price": -1},
        {"price": 150},
        {"month": 0},
        {"month": 13},
        {"n_screenshots": -3},
        {"genres": []},
        {"genres": ["Indie", "Farming"]},
        {"categories": ["Steam Trading Cards"]},
        {"unknown_field": 1},
    ],
)
def test_invalid_input_is_rejected(bad):
    with pytest.raises(ValidationError):
        Game(**bad)


def test_unknown_labels_are_named_in_the_error():
    with pytest.raises(ValidationError, match="Farming"):
        Game(genres=["Farming"])


def test_duplicate_labels_are_removed_in_order():
    assert Game(genres=["RPG", "Indie", "RPG"]).genres == ["RPG", "Indie"]


def test_self_published_game_shares_the_studio_history():
    game = Game(self_published=True, dev_prior_games=3, dev_prior_best_reviews=900, pub_prior_games=50)
    assert game.pub_prior_games == 3
    assert game.pub_prior_best_reviews == 900


def test_published_game_keeps_its_publisher_history():
    game = Game(self_published=False, dev_prior_games=3, pub_prior_games=50, pub_prior_best_reviews=7000)
    assert (game.pub_prior_games, game.pub_prior_best_reviews) == (50, 7000)


def test_first_game_has_no_previous_reviews():
    assert Game(dev_prior_games=0, dev_prior_best_reviews=5000).dev_prior_best_reviews == 0


def test_about_words_counts_the_description():
    assert Game(about="one two  three\nfour").about_words == 4


def test_with_changes_revalidates():
    game = Game(self_published=True, dev_prior_games=2)
    published = game.with_changes(self_published=False, pub_prior_games=40)
    assert published.pub_prior_games == 40
    assert game.self_published
    with pytest.raises(ValidationError):
        game.with_changes(price=-5)


def test_frame_matches_the_training_schema():
    frame = to_frame([Game(about="a b c"), Game(price=0)])
    assert len(frame) == 2
    assert frame["year"].eq(config.YEAR_MAX).all()
    assert frame["about_words"].tolist() == [3, 0]
