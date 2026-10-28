"""Text for the API reference pages."""

from gamesense import config
from gamesense.predictor import MIN_DESCRIPTION_WORDS
from gamesense.reliability import CLEAR_LEAD, HIGH_CONFIDENCE, MIN_CALIBRATION_GAMES
from gamesense.suggestions import MIN_REVIEW_RATIO, MIN_UPLIFT

TEST_YEAR = config.TEST_YEARS[0]
TRAIN_YEARS = f"{config.YEAR_MIN} to {config.YEAR_MAX}"

_tier_rows = "\n".join(f"| {tier} | {config.TIER_BLURBS[tier]} |" for tier in config.TIER_NAMES)

DESCRIPTION = f"""
Predicts how a game is likely to perform on Steam from information available before launch: genres, price,
platforms, Steam features, store page setup and the studio's track record. The model was trained on Steam
releases from {TRAIN_YEARS} and evaluated on {TEST_YEAR} releases it did not see during training.

The interactive console at [/docs](/docs) sends real requests from the browser.
Source code: [github.com/Dreem-R/gamesense-steam-predictor](https://github.com/Dreem-R/gamesense-steam-predictor).

## Quickstart

Send a game as JSON. Fields you leave out use their defaults.

```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"genres": ["Indie", "Action"], "price": 2.99, "n_screenshots": 5}}'
```

The response gives the most likely outcome tier, the chance of reaching 100+ reviews, a review range and checks
on how far to trust the prediction. `/explain` and `/what-if` take the same body.

## Outcome tiers

Total Steam reviews are the usual public stand-in for sales, at roughly 30 to 60 copies per review.

| Tier | Meaning |
|---|---|
{_tier_rows}

## Errors

Invalid input returns `422` with one entry per problem. The last item of `loc` names the field.

```json
{{
  "detail": [
    {{
      "type": "greater_than_equal",
      "loc": ["body", "price"],
      "msg": "Input should be greater than or equal to 0",
      "input": -5,
      "ctx": {{"ge": 0.0}}
    }}
  ]
}}
```

Unknown fields are rejected rather than ignored, so a misspelt field name does not silently fall back to its
default.

## Limits

- The model sees store page data only. Trailers, wishlists, marketing, press coverage and the quality of the
  game itself are not in the data.
- Reviews measure reach, not revenue: a $3 game and a $30 game with the same review count earned very
  different amounts.
- Store pages change after launch, so features such as languages and screenshots are somewhat optimistic.
- Effects are associations learned from past launches, not causes.
"""

TAGS = [
    {"name": "Predictions", "description": "Score a game described before launch."},
    {"name": "Service", "description": "Status and model information."},
]

HEALTH = "Returns `ok` once the model and reference data have loaded. Used by container health checks."

MODEL = (
    "Training set size, outcome tiers, number of features and the final model's scores on the "
    f"held-out {TEST_YEAR} releases."
)

PREDICT = f"""
Predicts the outcome tier, the chance of 100+ reviews and a review range for one game.

`reliability` compares the prediction with how the model did on held-out {TEST_YEAR} releases:

- `confidence` is High when one tier has at least {HIGH_CONFIDENCE:.0%} probability, Moderate when the top tier
  leads the next by {CLEAR_LEAD * 100:.0f} points or more, and Low otherwise.
- `tier_accuracy` is the share of held-out games placed in the same tier that ended up in it.
- `similar_predictions_rate` is the share of held-out games given a similar chance of 100+ reviews that
  reached it. It is `null` when fewer than {MIN_CALIBRATION_GAMES} such games exist.

`warnings` lists inputs that are rare in the training data, where predictions are less reliable.
`used_description` is false when `short_description` and `about` have fewer than {MIN_DESCRIPTION_WORDS} words
together, in which case the text model is skipped.
"""

EXPLAIN = """
Lists the features that move expected reviews the most for this game, compared with an average game.

Effects are SHAP values from the median review model, in log reviews. `multiplier` gives the same effect on
the review count: 1.5 means 50% more expected reviews, 0.7 means 30% fewer. Genre flags are summed into a
single `genres` entry, and release year is left out because it is fixed to the latest training year.
"""

WHAT_IF = f"""
Re-scores the game with each applicable change on its own, best first.

A change `helps` when it adds at least {MIN_UPLIFT * 100:.0f} point to the chance of 100+ reviews, or at least
{(MIN_REVIEW_RATIO - 1) * 100:.0f}% to expected reviews without lowering that chance. Only changes that apply are
tested: a game that already has Steam Cloud saves is not offered them.

`combined` applies every helpful low and medium effort change together and scores the result as one game,
because the effects of separate changes do not simply add up. It is `null` when fewer than two changes helped.
"""
