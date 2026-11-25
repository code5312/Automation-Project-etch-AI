"""ADR-0010/W9-06: `retrain/execute.py::current_champion_ckpt` 단위 테스트.

`DriftState`(`tests/unit/test_execute.py`)와 달리 `data/drift/raw`에 의존하지 않는다
— 레지스트리(`registry/`)만 읽으므로 별도 파일로 분리해 항상(로컬 drift 데이터 유무와
무관하게) 돌게 한다. torch는 모듈 import 시점에 당겨오지만(기존 `execute.py`가 이미
그랬음) 실제 모델 로드는 하지 않는다."""

import pathlib

import pytest

from etchpipe.gate.gate import GateResult
from etchpipe.registry.registry import Registry
from etchpipe.retrain import execute


def test_current_champion_ckpt_falls_back_when_no_champion_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(execute, "REPO_ROOT", tmp_path)  # registry/ 빈 디렉터리

    assert execute.current_champion_ckpt() == execute.CHAMPION_CKPT


def test_current_champion_ckpt_resolves_legacy_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(execute, "REPO_ROOT", tmp_path)
    reg = Registry(tmp_path / "registry")
    reg.promote("mgn_tuned_ep2", GateResult(passed=True, exit_code=0, reasons=[]))

    assert execute.current_champion_ckpt() == execute.CHAMPION_CKPT


def test_current_champion_ckpt_resolves_retrain_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(execute, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(execute, "RETRAIN_RUNS_ROOT", tmp_path / "runs" / "retrain")
    ckpt_dir = tmp_path / "runs" / "retrain" / "mgn_retrain_20261112T212130"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_bytes(b"fake")

    reg = Registry(tmp_path / "registry")
    reg.promote("mgn_retrain_20261112T212130", GateResult(passed=True, exit_code=0, reasons=[]))

    assert execute.current_champion_ckpt() == ckpt_path


def test_current_champion_ckpt_raises_when_artifact_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(execute, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(execute, "RETRAIN_RUNS_ROOT", tmp_path / "runs" / "retrain")
    reg = Registry(tmp_path / "registry")
    reg.promote("model-with-no-artifact-anywhere", GateResult(passed=True, exit_code=0, reasons=[]))

    with pytest.raises(RuntimeError, match="아티팩트를 찾을 수 없음"):
        execute.current_champion_ckpt()
