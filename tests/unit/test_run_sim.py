"""W7-03 실패 시나리오: `run_sim()`이 시뮬레이션 중간에 실패했을 때 뭘 남기는지 확인.

`run_sim()` 자체는 예외를 삼키지 않는다 — 호출부(`sim_batch.py`)가 재시도 정책을
결정하는 구조라 의도적으로 그렇게 둔다[확인, 코드 읽고 판단, 2026-11-14]. 실패 시
남는 상태(meta.json 없음, 부분 스냅샷만 있음)를 `etl.validate_run`이 크래시 없이
처리하는지는 `tests/unit/test_etl.py::test_validate_run_handles_missing_meta_json_
without_crashing`에서 이미 검증한다(그쪽은 `viennaps` 없이도 같은 디스크 상태를
직접 구성). `viennaps`가 필요해 이 파일은 `etch-sim:0.1` 컨테이너 전용."""

import pathlib

import pytest
import viennaps as ps

from etchpipe.sim.run_sim import run_sim

_PARAMS = {
    "ion_flux": 10.0, "etchant_flux": 5000.0, "oxygen_flux": 800.0,
    "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0,
}


def test_run_sim_succeeds_and_writes_meta_on_normal_path(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "ok_run"
    meta = run_sim(_PARAMS, out, K=2, rays=20)  # rays 작게: 테스트 속도용

    assert (out / "meta.json").exists()
    assert meta["actual_k"] == 2
    assert len(list((out / "snapshots").glob("t*.vtp"))) == meta["actual_k"] + 1


def test_run_sim_leaves_partial_snapshots_and_no_meta_when_process_fails(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    out = tmp_path / "crashed_run"
    calls = {"n": 0}
    real_apply = ps.Process.apply

    def _apply_then_fail(self):
        calls["n"] += 1
        if calls["n"] >= 2:  # t000(초기) 저장 후, 두 번째 스텝 진행 중 실패
            raise RuntimeError("synthetic ViennaPS failure for W7-03 test")
        return real_apply(self)

    monkeypatch.setattr(ps.Process, "apply", _apply_then_fail)

    with pytest.raises(RuntimeError, match="synthetic ViennaPS failure"):
        run_sim(_PARAMS, out, K=5, rays=20)

    # run_sim은 실패를 삼키지 않는다 -> meta.json은 안 쓰였어야 함
    assert not (out / "meta.json").exists()
    # 하지만 실패 전까지 쓴 스냅샷(t000.vtp 최소 1개)은 디스크에 남는다
    snaps = list((out / "snapshots").glob("t*.vtp"))
    assert len(snaps) >= 1
