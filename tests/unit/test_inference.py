"""W7-01 부수 수정: `load_champion`의 체크포인트 경로 해석(`_artifact_path_for`) 단위
테스트. W6-05/06 실행으로 champion이 `mgn_tuned_ep2`(고정 매핑)가 아닌
`mgn_retrain_20261112T212130`(재학습 산출물)로 바뀌면서 실제로 `/predict`가 503을
내던 버그의 회귀 테스트 — torch/physicsnemo는 안 쓴다(이 함수는 경로 계산만).

W7-03(실패 시나리오): `load_champion()` 자체의 세 가지 실패 경로(챔피언 없음, 경로
모름, 무결성 검증 실패)를 레지스트리 레이어에서 직접 검증한다 — 기존
`tests/unit/test_serve.py`는 이걸 `app.load_champion`을 통째로 모킹해서 HTTP 응답
(503)만 봤지, `inference.load_champion`의 레지스트리 조회·검증 로직 자체는 격리
테스트한 적이 없었다."""

import pytest

from etchpipe.gate.gate import GateResult
from etchpipe.registry.registry import Registry
from etchpipe.serve.inference import _ARTIFACT_PATHS, ChampionUnavailable, _artifact_path_for


def test_legacy_model_id_uses_fixed_mapping() -> None:
    assert _artifact_path_for("mgn_tuned_ep2") == _ARTIFACT_PATHS["mgn_tuned_ep2"]


def test_retrain_model_id_resolves_under_runs_retrain(monkeypatch, tmp_path) -> None:
    ckpt_dir = tmp_path / "mgn_retrain_20261112T212130"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_bytes(b"fake")
    monkeypatch.setattr("etchpipe.serve.inference._RETRAIN_RUNS_ROOT", tmp_path)

    assert _artifact_path_for("mgn_retrain_20261112T212130") == ckpt_path


def test_unknown_model_id_without_checkpoint_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("etchpipe.serve.inference._RETRAIN_RUNS_ROOT", tmp_path)

    assert _artifact_path_for("mgn_retrain_does_not_exist") is None


@pytest.fixture(autouse=True)
def _clear_champion_cache(monkeypatch):
    """`load_champion()`은 프로세스 전역 `_cache`에 캐시한다 — 테스트마다 빈 캐시로
    시작해야 서로 레지스트리 상태가 섞이지 않는다."""
    import etchpipe.serve.inference as inference_mod

    monkeypatch.setattr(inference_mod, "_cache", {})


def _empty_registry(tmp_path):
    return Registry(tmp_path / "registry")


# _validate_card(inference.py, PR#9 "W7-03 챔피언 로드 실패와 서빙 복구 보완"에서 추가)가
# THETA_KEYS 전부에 valid_range[key]["min"/"max"]를 요구한다 — 빈 dict로 register()하면
# load_champion()이 KeyError를 ChampionUnavailable로 삼켜버려 캐싱 테스트 자체가 실패한다.
_VALID_RANGE_FIXTURE = {
    key: {"min": 0.0, "max": 1.0}
    for key in [
        "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
        "ion_exponent", "mask_width", "mask_height",
    ]
}


def test_load_champion_raises_when_no_champion_registered(monkeypatch, tmp_path) -> None:
    from etchpipe.serve import inference

    monkeypatch.setattr(inference, "REGISTRY_ROOT", tmp_path / "registry")
    _empty_registry(tmp_path)  # root만 만들고 current는 안 씀 — "등록 전" 상태

    with pytest.raises(ChampionUnavailable, match="등록된 champion이 없습니다"):
        inference.load_champion()


def test_load_champion_raises_when_artifact_path_unknown(monkeypatch, tmp_path) -> None:
    from etchpipe.serve import inference

    monkeypatch.setattr(inference, "REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(inference, "_RETRAIN_RUNS_ROOT", tmp_path / "runs_retrain_empty")
    monkeypatch.setattr(inference, "_ARTIFACT_PATHS", {})

    reg = _empty_registry(tmp_path)
    # register()는 artifact_path로 실제 파일을 요구(sha256 계산) — 임시 파일로 등록만
    # 해 두고, load_champion이 "다음에 어디서 찾을지 모르는" model_id를 만나게 한다.
    fake_ckpt = tmp_path / "fake_model.pt"
    fake_ckpt.write_bytes(b"fake-checkpoint-bytes")
    reg.register(
        "model-with-no-known-path", fake_ckpt, GateResult(passed=True, exit_code=0),
        dataset_hash="deadbeef0000", config={}, git_sha="unknown", metrics={}, valid_range={},
    )

    with pytest.raises(ChampionUnavailable, match="체크포인트를 찾을 수 없습니다"):
        inference.load_champion()


def test_load_champion_raises_when_artifact_tampered(monkeypatch, tmp_path) -> None:
    from etchpipe.serve import inference

    monkeypatch.setattr(inference, "REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(inference, "_ARTIFACT_PATHS", {})
    monkeypatch.setattr(inference, "_RETRAIN_RUNS_ROOT", tmp_path / "runs_retrain")

    reg = _empty_registry(tmp_path)
    model_id = "mgn_retrain_tampered_test"
    ckpt_path = tmp_path / "runs_retrain" / model_id / "model.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"original-checkpoint-bytes")
    reg.register(
        model_id, ckpt_path, GateResult(passed=True, exit_code=0),
        dataset_hash="deadbeef0000", config={}, git_sha="unknown", metrics={}, valid_range={},
    )

    ckpt_path.write_bytes(b"tampered-after-registration")  # 등록 후 바이트 변조

    with pytest.raises(ChampionUnavailable, match="무결성 검증 실패"):
        inference.load_champion()


def test_load_champion_caches_result_and_does_not_reverify(monkeypatch, tmp_path) -> None:
    from etchpipe.serve import inference

    monkeypatch.setattr(inference, "REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(inference, "_ARTIFACT_PATHS", {})
    monkeypatch.setattr(inference, "_RETRAIN_RUNS_ROOT", tmp_path / "runs_retrain")

    reg = _empty_registry(tmp_path)
    model_id = "mgn_retrain_cache_test"
    ckpt_path = tmp_path / "runs_retrain" / model_id / "model.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"checkpoint-bytes")
    reg.register(
        model_id, ckpt_path, GateResult(passed=True, exit_code=0),
        dataset_hash="deadbeef0000", config={}, git_sha="unknown", metrics={},
        valid_range=_VALID_RANGE_FIXTURE,
    )

    calls = {"n": 0}

    def _fake_load_model_a(path):
        calls["n"] += 1
        return "fake-model", "fake-note", None

    monkeypatch.setattr("etchpipe.eval.eval.load_model_a", _fake_load_model_a)

    first = inference.load_champion()
    second = inference.load_champion()

    assert first is second  # 같은 캐시 딕셔너리 객체
    assert calls["n"] == 1  # load_model_a는 최초 1회만
