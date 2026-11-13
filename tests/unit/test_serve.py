"""W2-08/W6-08: 서빙(/health, /model, /predict, API 키, 입력 검증, 요청 크기·속도
제한) 단위 테스트. 실모델/ViennaPS는 여기서 안 쓴다 — `load_champion`/`predict_recipe`를
모킹해 HTTP 레이어(인증·검증·레이트리밋·바디크기)만 검사한다(이 둘은 서빙 전용
컨테이너(`etch-serve:0.1`)에서만 실제로 동작, train 컨테이너에는 없음)."""

import json
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


def test_predict_mask_width_above_ceiling_is_422() -> None:
    # W6-09: mask_width가 극단적으로 크면(실측: ViennaPS 메시 생성이 멈추거나 죽음)
    # 추론 단계까지 못 가고 여기서 막혀야 한다.
    bad_recipe = dict(_RECIPE, mask_width=1e9)
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def test_predict_flux_above_ceiling_is_422() -> None:
    bad_recipe = dict(_RECIPE, ion_flux=1e12)
    resp = client.post(
        "/predict", json={"recipes": [bad_recipe], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422


def _post_raw(body_text: str):
    return client.post(
        "/predict", content=body_text,
        headers={"x-api-key": API_KEY, "content-type": "application/json"},
    )


def test_predict_nan_field_is_422_not_500() -> None:
    # W6-09 퍼징으로 발견: gt=0/le=...만으로는 NaN을 못 걸러내 실제 추론까지 흘러갔다
    # (출력이 NaN이 돼 JSON 직렬화 단계에서 500). allow_inf_nan=False로 입력에서 막는다.
    body = '{"recipes": [{"ion_flux": NaN, "etchant_flux": 4500.0, "oxygen_flux": 800.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0}], "times": [0.5]}'  # noqa: E501
    resp = _post_raw(body)
    assert resp.status_code == 422


def test_predict_infinity_field_is_422_not_500() -> None:
    body = '{"recipes": [{"ion_flux": Infinity, "etchant_flux": 4500.0, "oxygen_flux": 800.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0}], "times": [0.5]}'  # noqa: E501
    resp = _post_raw(body)
    assert resp.status_code == 422


def test_predict_nan_time_is_422() -> None:
    body = '{"recipes": [{"ion_flux": 10.0, "etchant_flux": 4500.0, "oxygen_flux": 800.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0}], "times": [NaN]}'  # noqa: E501
    resp = _post_raw(body)
    assert resp.status_code == 422


def test_predict_unexpected_inference_error_is_422_not_500(monkeypatch) -> None:
    # W6-09: Field 상한을 다 통과했지만 추론 단계에서만 드러나는 오류는 422로 돌리고
    # 절대 5xx를 내지 않는다. 에러 메시지에 경로·스택이 안 보여야 한다.
    def _boom(theta_dict, times):
        raise RuntimeError("/app/some/internal/path.py line 42 boom")

    monkeypatch.setattr("etchpipe.serve.app.predict_recipe", _boom)
    resp = client.post(
        "/predict", json={"recipes": [_RECIPE], "times": [0.5]}, headers={"x-api-key": API_KEY}
    )
    assert resp.status_code == 422
    assert "/app/" not in resp.text
    assert "Traceback" not in resp.text


def test_model_unexpected_error_falls_back_to_sanitized_500(monkeypatch) -> None:
    # /model은 ChampionUnavailable만 따로 잡는다 — 그 외 예외는 전역 핸들러(
    # ServerErrorMiddleware)가 받아서 경로·스택 없이 500으로 바꿔야 한다(2차 방어막).
    # 기본 TestClient(raise_server_exceptions=True)는 테스트 편의상 이 핸들러를 건너
    # 뛰고 예외를 그대로 올리므로, 이 테스트만 raise_server_exceptions=False로 돌려
    # 실제 배포(uvicorn)와 같은 응답을 확인한다.
    def _boom():
        raise RuntimeError("/app/some/internal/path.py line 99 kaboom")

    monkeypatch.setattr("etchpipe.serve.app.load_champion", _boom)
    lenient_client = TestClient(app, raise_server_exceptions=False)
    resp = lenient_client.get("/model", headers={"x-api-key": API_KEY})
    assert resp.status_code == 500
    assert "/app/" not in resp.text
    assert "Traceback" not in resp.text


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


# W7-01: 대시보드용 공개 엔드포인트(`/api/predict`, `/api/models/compare`,
# `/api/gate/history`) — 브라우저가 API 키 없이 쓰는 경로라 `/predict`와 달리
# 헤더 없이 호출돼야 한다.


def test_api_predict_without_api_key_returns_matching_shape(monkeypatch) -> None:
    monkeypatch.setattr("etchpipe.serve.app.predict_recipe", _fake_predict_recipe)
    resp = client.post("/api/predict", json={"recipes": [_RECIPE], "times": [0.5]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "fake-champion"
    assert len(body["results"]) == 1


def test_api_predict_missing_field_is_422() -> None:
    bad_recipe = dict(_RECIPE)
    del bad_recipe["ion_flux"]
    resp = client.post("/api/predict", json={"recipes": [bad_recipe], "times": [0.5]})
    assert resp.status_code == 422


def test_api_models_compare_returns_file_contents(monkeypatch, tmp_path) -> None:
    payload = {"summary": {"n": 1, "A_chamfer_mean": 0.1}, "rows": []}
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr("etchpipe.serve.app.MODEL_COMPARISON_PATH", path)
    resp = client.get("/api/models/compare")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_api_models_compare_returns_503_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("etchpipe.serve.app.MODEL_COMPARISON_PATH", tmp_path / "missing.json")
    resp = client.get("/api/models/compare")
    assert resp.status_code == 503


def test_api_gate_history_returns_entries(monkeypatch, tmp_path) -> None:
    entry = {"action": "promote", "model_id": "m1", "promoted": True, "exit_code": 0, "reasons": []}
    (tmp_path / "history.jsonl").write_text(json.dumps(entry) + "\n")
    monkeypatch.setattr("etchpipe.serve.app.REGISTRY_ROOT", tmp_path)
    resp = client.get("/api/gate/history")
    assert resp.status_code == 200
    assert resp.json() == {"entries": [entry]}


def test_api_gate_history_empty_when_no_history_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("etchpipe.serve.app.REGISTRY_ROOT", tmp_path)
    resp = client.get("/api/gate/history")
    assert resp.status_code == 200
    assert resp.json() == {"entries": []}


def test_dashboard_index_served_at_root() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "EtchGuard" in resp.text or "식각" in resp.text


def test_dashboard_unknown_path_is_404_not_predict_route() -> None:
    resp = client.get("/no-such-dashboard-page")
    assert resp.status_code == 404
