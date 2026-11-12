"""W2-08/W6-08: 서빙(/health, /model, /predict, API 키, 입력 검증, 요청 크기·속도
제한) 단위 테스트. 실모델/ViennaPS는 여기서 안 쓴다 — `load_champion`/`predict_recipe`를
모킹해 HTTP 레이어(인증·검증·레이트리밋·바디크기)만 검사한다(이 둘은 서빙 전용
컨테이너(`etch-serve:0.1`)에서만 실제로 동작, train 컨테이너에는 없음)."""

import os

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient as StarletteTestClient

from etchpipe.serve.app import _BodySizeLimitMiddleware, _RateLimitMiddleware, app
from etchpipe.serve.inference import ChampionUnavailable

client = TestClient(app)
API_KEY = os.environ["ETCH_API_KEY"]

_RECIPE = {
    "ion_flux": 10.0, "etchant_flux": 5000.0, "oxygen_flux": 800.0,
    "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0,
}

_FAKE_CARD = {
    "valid_range": {"ion_flux": {"min": 3.0, "max": 30.0}},
    "gate_result": {"passed": True, "exit_code": 0, "reasons": []},
}
_FAKE_CHAMPION = {"model_id": "fake-champion", "card": _FAKE_CARD}


def _fake_predict_recipe(theta_dict: dict, times: list[float]) -> dict:
    return {
        "model_version": "fake-champion",
        "profiles": [{"t": t, "points": [[0.0, 0.0]]} for t in times],
        "cd": 1.0,
        "depth": 2.0,
        "taper_deg": 3.0,
        "in_range": True,
        "clamped": False,
    }


def test_health_requires_no_api_key() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_without_api_key_is_401() -> None:
    resp = client.get("/model")
    assert resp.status_code == 401


def test_model_with_wrong_api_key_is_401() -> None:
    """W3-10: hmac.compare_digest로 교체 후에도 틀린 키는 여전히 401이어야 한다
    (상수시간 비교로 바뀌어도 판정 결과는 동일해야 함)."""
    resp = client.get("/model", headers={"x-api-key": API_KEY + "x"})
    assert resp.status_code == 401


def test_model_with_api_key_returns_real_registry_shape(monkeypatch) -> None:
    monkeypatch.setattr("etchpipe.serve.app.load_champion", lambda: _FAKE_CHAMPION)
    resp = client.get("/model", headers={"x-api-key": API_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "fake-champion"
    assert body["valid_range"] == _FAKE_CARD["valid_range"]
    assert body["gate_result"] == _FAKE_CARD["gate_result"]


def test_model_returns_503_when_champion_unavailable(monkeypatch) -> None:
    def _raise():
        raise ChampionUnavailable("no champion")

    monkeypatch.setattr("etchpipe.serve.app.load_champion", _raise)
    resp = client.get("/model", headers={"x-api-key": API_KEY})
    assert resp.status_code == 503


def test_predict_without_api_key_is_401() -> None:
    resp = client.post("/predict", json={"recipes": [_RECIPE], "times": [0.5]})
    assert resp.status_code == 401


def test_predict_with_api_key_returns_matching_shape(monkeypatch) -> None:
    monkeypatch.setattr("etchpipe.serve.app.predict_recipe", _fake_predict_recipe)
    resp = client.post(
        "/predict",
        json={"recipes": [_RECIPE, _RECIPE], "times": [0.25, 0.5]},
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "fake-champion"
    assert len(body["results"]) == 2
    assert len(body["results"][0]["profiles"]) == 2
    assert body["results"][0]["in_range"] is True


def test_predict_returns_503_when_champion_unavailable(monkeypatch) -> None:
    def _raise(theta_dict, times):
        raise ChampionUnavailable("no champion")

    monkeypatch.setattr("etchpipe.serve.app.predict_recipe", _raise)
    resp = client.post(
        "/predict", json={"recipes": [_RECIPE], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 503


def test_predict_missing_field_is_422() -> None:
    bad_recipe = dict(_RECIPE)
    del bad_recipe["ion_flux"]
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def test_predict_negative_value_is_422() -> None:
    bad_recipe = dict(_RECIPE, ion_flux=-1.0)
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def test_predict_zero_value_is_422() -> None:
    bad_recipe = dict(_RECIPE, mask_width=0.0)
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def test_predict_too_many_recipes_is_422() -> None:
    from etchpipe.serve.app import MAX_BATCH_RECIPES

    resp = client.post(
        "/predict",
        json={"recipes": [_RECIPE] * (MAX_BATCH_RECIPES + 1), "times": [0.5]},
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 422


def test_predict_too_many_times_is_422() -> None:
    from etchpipe.serve.app import MAX_TIMES

    resp = client.post(
        "/predict",
        json={"recipes": [_RECIPE], "times": [0.5] * (MAX_TIMES + 1)},
        headers={"x-api-key": API_KEY},
    )
    assert resp.status_code == 422


def test_predict_non_positive_time_is_422() -> None:
    resp = client.post(
        "/predict", json={"recipes": [_RECIPE], "times": [0.0]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def test_predict_empty_recipes_is_422() -> None:
    resp = client.post(
        "/predict", json={"recipes": [], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def _tiny_middleware_app(middleware_cls, **kwargs) -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    a = Starlette(routes=[Route("/x", ok, methods=["GET", "POST"])])
    a.add_middleware(middleware_cls, **kwargs)
    return a


def test_body_size_limit_middleware_rejects_oversized_content_length() -> None:
    a = _tiny_middleware_app(_BodySizeLimitMiddleware)
    tc = StarletteTestClient(a)
    resp = tc.post("/x", headers={"content-length": "999999999"}, content=b"x")
    assert resp.status_code == 413


def test_body_size_limit_middleware_allows_small_requests() -> None:
    a = _tiny_middleware_app(_BodySizeLimitMiddleware)
    tc = StarletteTestClient(a)
    resp = tc.get("/x")
    assert resp.status_code == 200


def test_rate_limit_middleware_blocks_after_limit_exceeded() -> None:
    a = _tiny_middleware_app(_RateLimitMiddleware, limit_per_minute=2)
    tc = StarletteTestClient(a)
    assert tc.get("/x").status_code == 200
    assert tc.get("/x").status_code == 200
    assert tc.get("/x").status_code == 429


def test_rate_limit_middleware_tracks_separate_keys_independently() -> None:
    a = _tiny_middleware_app(_RateLimitMiddleware, limit_per_minute=1)
    tc = StarletteTestClient(a)
    assert tc.get("/x", headers={"x-api-key": "key-a"}).status_code == 200
    assert tc.get("/x", headers={"x-api-key": "key-b"}).status_code == 200
    assert tc.get("/x", headers={"x-api-key": "key-a"}).status_code == 429
