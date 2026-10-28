import subprocess
import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from gamesense import config
from gamesense.api.main import app
from gamesense.game import Game
from gamesense.presets import PRESETS

NOVEL = PRESETS["Story-driven visual novel"]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_the_reference(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/redoc"


def test_model_info(client, predictor):
    info = client.get("/model").json()
    assert info["train_rows"] == predictor.model.train_rows
    assert info["tiers"] == config.TIER_NAMES
    assert info["test_year"] == config.TEST_YEARS[0]
    assert 0.5 < info["test_metrics"]["auc_success_plus"] <= 1


def test_predict_matches_the_library(client, predictor, preset_game):
    body = client.post("/predict", json=preset_game.model_dump()).json()
    pred = predictor.predict(preset_game)
    assert body["tier"] == pred.tier_name
    assert list(body["probabilities"]) == config.TIER_NAMES
    assert body["success_chance"] == pytest.approx(pred.success_chance)
    assert body["reviews"]["expected"] == pytest.approx(pred.reviews_mid)
    assert body["reliability"]["confidence"] in {"High", "Moderate", "Low"}


def test_predict_fills_defaults_for_missing_fields(client, predictor):
    body = client.post("/predict", json={"genres": ["Action"]}).json()
    assert body["success_chance"] == pytest.approx(predictor.predict(Game(genres=["Action"])).success_chance)


def test_predict_reports_unusual_inputs(client):
    body = client.post("/predict", json=NOVEL.model_dump() | {"about": "", "price": 79.99}).json()
    assert body["used_description"] is False
    assert any("$60" in w for w in body["warnings"])


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"genres": ["Farming"]}, "genres"),
        ({"genres": []}, "genres"),
        ({"price": -1}, "price"),
        ({"month": 13}, "month"),
        ({"n_screenshots": "many"}, "n_screenshots"),
        ({"wishlists": 5000}, "wishlists"),
    ],
)
def test_invalid_games_are_rejected_with_the_field_named(client, payload, field):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert [e["loc"][-1] for e in response.json()["detail"]] == [field]


def test_explain_matches_the_library(client, predictor):
    body = client.post("/explain", params={"top_k": 3}, json=NOVEL.model_dump()).json()
    expected = predictor.explain(NOVEL, top_k=3)
    assert [c["feature"] for c in body] == [c.feature for c in expected]
    assert [c["effect"] for c in body] == pytest.approx([c.effect for c in expected])


@pytest.mark.parametrize("top_k", [0, 31])
def test_explain_limits_top_k(client, top_k):
    assert client.post("/explain", params={"top_k": top_k}, json=NOVEL.model_dump()).status_code == 422


def test_what_if_matches_the_library(client, predictor):
    body = client.post("/what-if", json=NOVEL.model_dump()).json()
    scored, plan = predictor.what_if(NOVEL)
    assert [c["id"] for c in body["changes"]] == [s.change.id for s in scored]
    assert [c["helps"] for c in body["changes"]] == [s.helps for s in scored]
    assert plan is not None
    assert body["combined"]["change_ids"] == plan.change_ids
    assert body["combined"]["success_after"] == pytest.approx(plan.success_after)


def test_api_serves_predictions_from_a_fresh_process():
    """Imports the app the way uvicorn does, without the modules this test session already loaded."""
    code = (
        "import uvicorn\n"
        "from fastapi.testclient import TestClient\n"
        "from gamesense.api.main import app\n"
        "with TestClient(app) as c:\n"
        "    r = c.post('/predict', json={'genres': ['Indie']})\n"
        "    assert r.status_code == 200, r.text\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]


def test_every_endpoint_is_documented(client):
    spec = client.get("/openapi.json").json()
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            assert op.get("tags"), (method, path)
            assert op.get("summary"), (method, path)
            assert len(op.get("description", "")) > 40, (method, path)


def test_reference_lists_allowed_values(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert schemas["Game"]["properties"]["genres"]["items"]["enum"] == config.GENRES
    assert "steam_cloud" in schemas["ScoredChange"]["properties"]["id"]["enum"]


def test_docs_offer_the_example_games(client):
    spec = client.get("/openapi.json").json()
    examples = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert {e["summary"] for e in examples.values()} == set(PRESETS) - {"Start from scratch"}
