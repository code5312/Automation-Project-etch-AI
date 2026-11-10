"""W5-02: 평가 진입점(`eval.run_eval`) 단위 테스트.

실제 MeshGraphNet(physicsnemo 의존)은 호스트에 없을 수 있어 토이 모델로 집계·플롯·구조만
검증한다(모델 연결 자체는 W4-03/04/06 증거로 이미 확인됨, `load_model_a`도 같은 이유)."""

import json
import pathlib
import subprocess

import pytest
import torch
from torch import nn

import etchpipe.eval.eval as eval_mod
from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.eval.eval import _agg, hardware_info, run_eval, safe_git_sha

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"
RUN_IDS = ["r0000", "r0001", "r0002"]


class ToyNodeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(14, 2)

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return self.lin(node)


def _build_graphs(tmp_path: pathlib.Path) -> pathlib.Path:
    graphs_root = tmp_path / "graphs"
    for rid in RUN_IDS:
        etl_run(FIXTURES / rid, graphs_root / rid, THETA_KEYS)
    return graphs_root


def test_agg_reports_mean_median_and_count() -> None:
    assert _agg([1.0, 2.0, 3.0]) == {"mean": 2.0, "median": 2.0, "n": 3}


def test_hardware_info_includes_device_and_platform() -> None:
    info = hardware_info("cpu")
    assert info["device"] == "cpu"
    assert "platform" in info
    assert "torch" in info


def test_safe_git_sha_falls_back_when_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> str:
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(eval_mod, "git_sha", _raise)
    assert safe_git_sha() == "[미확인: git 명령 실패]"


def test_safe_git_sha_falls_back_on_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> str:
        raise subprocess.CalledProcessError(1, "git rev-parse HEAD")

    monkeypatch.setattr(eval_mod, "git_sha", _raise)
    assert safe_git_sha() == "[미확인: git 명령 실패]"


def test_safe_git_sha_passes_through_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "git_sha", lambda: "abc123")
    assert safe_git_sha() == "abc123"


def test_run_eval_writes_metrics_structure_and_plots(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    out_dir = tmp_path / "eval_out"

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = run_eval(
        model, graphs_root, FIXTURES, RUN_IDS, norm=None, out_dir=out_dir, steps=(1, 2), n_plots=2
    )

    assert result["n_val"] == 3
    assert set(result["final_step"]) == {"chamfer", "depth_error"}
    assert set(result["by_step"]) <= {"1", "2"}
    assert len(result["per_run"]) == 3
    for entry in result["per_run"]:
        assert entry["actual_k"] == 3
        assert 3 in entry["steps"]  # 최종 스텝은 steps 목록에 없어도 항상 포함

    # 정렬된 순서로 앞 2개(r0000, r0001)만 플롯이 생겨야 함
    plot_files = sorted(p.name for p in (out_dir / "plots").glob("*.png"))
    assert plot_files == ["r0000.png", "r0001.png"]


def test_run_eval_requested_step_beyond_actual_k_is_skipped(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    out_dir = tmp_path / "eval_out"

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = run_eval(
        model, graphs_root, FIXTURES, RUN_IDS, norm=None, out_dir=out_dir, steps=(10,), n_plots=0
    )

    # actual_k=3인 fixture들에서 step=10은 존재할 수 없으므로 by_step에 안 잡힘
    assert "10" not in result["by_step"]
    assert "3" not in result["by_step"]  # 최종 스텝은 final_step 쪽으로만 집계
    for entry in result["per_run"]:
        assert set(entry["steps"]) == {3}  # 요청 스텝(10) 제외, 최종 스텝(3)만 남음


def test_run_eval_is_deterministic_across_repeated_calls(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)

    torch.manual_seed(0)
    model_a = ToyNodeMLP()
    result_a = run_eval(
        model_a, graphs_root, FIXTURES, RUN_IDS, norm=None,
        out_dir=tmp_path / "out_a", steps=(1, 2), n_plots=0,
    )

    torch.manual_seed(0)
    model_b = ToyNodeMLP()
    result_b = run_eval(
        model_b, graphs_root, FIXTURES, RUN_IDS, norm=None,
        out_dir=tmp_path / "out_b", steps=(1, 2), n_plots=0,
    )

    assert json.dumps(result_a, sort_keys=True) == json.dumps(result_b, sort_keys=True)
