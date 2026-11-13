"""W7-01 부수 수정: `load_champion`의 체크포인트 경로 해석(`_artifact_path_for`) 단위
테스트. W6-05/06 실행으로 champion이 `mgn_tuned_ep2`(고정 매핑)가 아닌
`mgn_retrain_20261112T212130`(재학습 산출물)로 바뀌면서 실제로 `/predict`가 503을
내던 버그의 회귀 테스트 — torch/physicsnemo는 안 쓴다(이 함수는 경로 계산만)."""

import json
import sys
from types import ModuleType

import pytest

from etchpipe.etl.etl import THETA_KEYS
from etchpipe.registry.registry import sha256_file
from etchpipe.serve import inference
from etchpipe.serve.inference import _ARTIFACT_PATHS, _artifact_path_for


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


@pytest.mark.parametrize(
    "failure",
    ["missing_checkpoint", "missing_card", "invalid_card", "bounds", "hash", "load"],
)
def test_unavailable_champion_can_be_retried(monkeypatch, tmp_path, failure) -> None:
    model_id = "mgn_tuned_ep2"
    registry = tmp_path / "registry"
    card_path = registry / model_id / "card.json"
    card_path.parent.mkdir(parents=True)
    (registry / "current").write_text(model_id)
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"checkpoint")
    card = {
        "artifact_sha256": sha256_file(artifact),
        "valid_range": {key: {"min": 1, "max": 2} for key in THETA_KEYS},
        "gate_result": None,
    }
    card_path.write_text(json.dumps(card))
    monkeypatch.setattr(inference, "_cache", {})
    monkeypatch.setattr(inference, "REGISTRY_ROOT", registry)
    monkeypatch.setitem(inference._ARTIFACT_PATHS, model_id, artifact)
    evaluator = ModuleType("etchpipe.eval.eval")
    model = object()

    def load_model(path):
        if path.read_bytes() != b"checkpoint":
            raise ValueError("Invalid weights")
        return model, "test model", None

    evaluator.load_model_a = load_model
    monkeypatch.setitem(sys.modules, evaluator.__name__, evaluator)

    if failure == "missing_checkpoint":
        artifact.unlink()
    elif failure == "missing_card":
        card_path.unlink()
    elif failure == "invalid_card":
        card_path.write_text("{")
    elif failure == "bounds":
        card_path.write_text(json.dumps({**card, "valid_range": {}}))
    elif failure == "hash":
        artifact.write_bytes(b"modified")
    else:
        artifact.write_bytes(b"unreadable")
        card_path.write_text(json.dumps({**card, "artifact_sha256": sha256_file(artifact)}))

    with pytest.raises(inference.ChampionUnavailable):
        inference.load_champion()
    assert inference._cache == {}

    artifact.write_bytes(b"checkpoint")
    card_path.write_text(json.dumps(card))
    assert inference.load_champion()["model"] is model
