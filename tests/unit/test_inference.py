"""W7-01 부수 수정: `load_champion`의 체크포인트 경로 해석(`_artifact_path_for`) 단위
테스트. W6-05/06 실행으로 champion이 `mgn_tuned_ep2`(고정 매핑)가 아닌
`mgn_retrain_20261112T212130`(재학습 산출물)로 바뀌면서 실제로 `/predict`가 503을
내던 버그의 회귀 테스트 — torch/physicsnemo는 안 쓴다(이 함수는 경로 계산만)."""

from etchpipe.serve.inference import _artifact_path_for, _ARTIFACT_PATHS


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
