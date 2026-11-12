"""W2-07/W5-07: 레지스트리(승격·champion 포인터·history.jsonl + 자동 모델 카드·
valid_range.json·manifest.json·SHA-256 무결성·롤백) 단위 테스트."""

import hashlib
import json
from pathlib import Path

import pytest

from etchpipe.gate.gate import GateResult
from etchpipe.registry.registry import Registry, compute_valid_range, sha256_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def test_promote_on_pass_sets_champion(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    ok = reg.promote("model-a", GateResult(passed=True, exit_code=0, reasons=[]))
    assert ok is True
    assert reg.current_champion() == "model-a"


def test_promote_on_fail_does_not_set_champion(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.promote("model-a", GateResult(passed=True, exit_code=0, reasons=[]))
    ok = reg.promote("model-b", GateResult(passed=False, exit_code=1, reasons=["bad"]))
    assert ok is False
    assert reg.current_champion() == "model-a"  # champion 유지


def test_history_records_every_attempt(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.promote("model-a", GateResult(passed=True, exit_code=0, reasons=[]))
    reg.promote("model-b", GateResult(passed=False, exit_code=1, reasons=["bad"]))
    hist = reg.history()
    assert len(hist) == 2
    assert hist[0]["model_id"] == "model-a"
    assert hist[0]["promoted"] is True
    assert hist[1]["model_id"] == "model-b"
    assert hist[1]["promoted"] is False
    assert hist[1]["exit_code"] == 1


def test_current_champion_none_before_any_promotion(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    assert reg.current_champion() is None


def test_sha256_file_matches_hashlib_reference(tmp_path: Path) -> None:
    f = tmp_path / "artifact.bin"
    f.write_bytes(b"checkpoint bytes" * 100)
    assert sha256_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_compute_valid_range_matches_min_max_across_runs() -> None:
    vr = compute_valid_range(FIXTURES, ["r0000", "r0001", "r0002"])
    assert vr["ion_flux"]["min"] == pytest.approx(3.4551301957147644)
    assert vr["ion_flux"]["max"] == pytest.approx(11.386263904649711)
    assert set(vr) == {
        "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
        "ion_exponent", "mask_width", "mask_height",
    }
    for rng in vr.values():
        assert rng["min"] <= rng["max"]


def _artifact(tmp_path: Path) -> Path:
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"fake checkpoint")
    return f


def test_register_on_pass_writes_card_and_promotes(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    artifact = _artifact(tmp_path)
    ok = reg.register(
        "model-a",
        artifact,
        GateResult(passed=True, exit_code=0, reasons=[]),
        dataset_hash="abc123",
        config={"lr": 0.0001},
        git_sha="deadbeef",
        metrics={"chamfer": {"median": 1.0}},
        valid_range={"ion_flux": {"min": 1.0, "max": 10.0}},
    )

    assert ok is True
    assert reg.current_champion() == "model-a"
    card = reg.load_card("model-a")
    assert card["dataset_hash"] == "abc123"
    assert card["artifact_sha256"] == sha256_file(artifact)
    assert (reg.root / "model-a" / "valid_range.json").exists()
    manifest = json.loads(reg.manifest_path.read_text())
    assert manifest["current"] == "model-a"
    assert manifest["models"] == ["model-a"]


def test_register_on_block_writes_no_card_but_logs_history(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    artifact = _artifact(tmp_path)
    ok = reg.register(
        "model-bad",
        artifact,
        GateResult(passed=False, exit_code=1, reasons=["golden chamfer too high"]),
        dataset_hash="abc123",
        config={},
        git_sha="deadbeef",
        metrics={},
        valid_range={},
    )

    assert ok is False
    assert reg.current_champion() is None
    assert not (reg.root / "model-bad").exists()  # 설계서 §10: BLOCK은 registry/에 안 씀
    hist = reg.history()
    assert len(hist) == 1
    assert hist[0]["model_id"] == "model-bad"
    assert hist[0]["promoted"] is False


def test_verify_artifact_detects_tampering(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    artifact = _artifact(tmp_path)
    reg.register(
        "model-a", artifact, GateResult(passed=True, exit_code=0, reasons=[]),
        dataset_hash="h", config={}, git_sha="g", metrics={}, valid_range={},
    )

    assert reg.verify_artifact("model-a", artifact) is True
    artifact.write_bytes(b"tampered bytes")
    assert reg.verify_artifact("model-a", artifact) is False


def test_rollback_to_previously_registered_model(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    artifact_a = _artifact(tmp_path)
    reg.register(
        "model-a", artifact_a, GateResult(passed=True, exit_code=0, reasons=[]),
        dataset_hash="h", config={}, git_sha="g", metrics={}, valid_range={},
    )
    artifact_b = tmp_path / "ckpt_b.pt"
    artifact_b.write_bytes(b"model b bytes")
    reg.register(
        "model-b", artifact_b, GateResult(passed=True, exit_code=0, reasons=[]),
        dataset_hash="h", config={}, git_sha="g", metrics={}, valid_range={},
    )
    assert reg.current_champion() == "model-b"

    reg.rollback("model-a")

    assert reg.current_champion() == "model-a"
    hist = reg.history()
    assert hist[-1]["action"] == "rollback"
    assert hist[-1]["model_id"] == "model-a"


def test_rollback_rejects_unregistered_model_id(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    with pytest.raises(ValueError):
        reg.rollback("never-registered")
