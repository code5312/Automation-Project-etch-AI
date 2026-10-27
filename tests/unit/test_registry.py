"""W2-07: 레지스트리 최소판(승격, champion 포인터, history.jsonl) 단위 테스트."""

from pathlib import Path

from etchpipe.gate.gate import GateResult
from etchpipe.registry.registry import Registry


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
