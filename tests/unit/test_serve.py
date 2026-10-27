"""W2-08: 서빙 최소판(/health, /model, /predict, API 키) 단위 테스트."""

import os

from fastapi.testclient import TestClient

from etchpipe.serve.app import app

client = TestClient(app)
API_KEY = os.environ["ETCH_API_KEY"]

_RECIPE = {
    "ion_flux": 10.0, "etchant_flux": 5000.0, "oxygen_flux": 800.0,
    "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0,
}


def test_health_requires_no_api_key() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_without_api_key_is_401() -> None:
    resp = client.get("/model")
    assert resp.status_code == 401


def test_model_with_api_key_is_200() -> None:
    resp = client.get("/model", headers={"x-api-key": API_KEY})
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_predict_without_api_key_is_401() -> None:
    resp = client.post("/predict", json={"recipes": [_RECIPE], "times": [0.5]})
    assert resp.status_code == 401


def test_predict_with_api_key_returns_matching_shape() -> None:
    resp = client.post(
        "/predict",
        json={"recipes": [_RECIPE, _RECIPE], "times": [0.25, 0.5]},
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert len(body["results"][0]["profiles"]) == 2


def test_predict_missing_field_is_422() -> None:
    bad_recipe = dict(_RECIPE)
    del bad_recipe["ion_flux"]
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422
