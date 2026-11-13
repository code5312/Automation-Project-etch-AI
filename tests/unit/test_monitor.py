"""W6-02: 드리프트 모니터 단위 테스트."""

import pathlib

import torch
from torch import nn

from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.retrain.monitor import check_drift, mean_chamfer_over_recipes

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"
RUN_IDS = ["r0000", "r0001", "r0002"]


class ZeroModel(nn.Module):
    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return torch.zeros((node.shape[0], 2))


def _build_graphs(tmp_path: pathlib.Path) -> pathlib.Path:
    graphs_root = tmp_path / "graphs"
    for rid in RUN_IDS:
        etl_run(FIXTURES / rid, graphs_root / rid, THETA_KEYS)
    return graphs_root


def test_mean_chamfer_over_recipes_reports_mean_and_per_recipe(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = _build_graphs(tmp_path)
    result = mean_chamfer_over_recipes(ZeroModel(), None, graphs_root, FIXTURES, RUN_IDS)

    assert set(result) == {"mean", "per_recipe"}
    assert set(result["per_recipe"]) == set(RUN_IDS)
    for v in result["per_recipe"].values():
        assert v >= 0.0
    assert result["mean"] >= 0.0


def test_check_drift_triggers_when_pass_rate_below_threshold(
    tmp_path: pathlib.Path,
) -> None:
    # ADR-0005 결정5: ZeroModel(항상 0 변위, 전혀 안 움직임)이 실제 식각된 정답과
    # 비교되면 깊이 오차가 당연히 임계치를 넘어 3조건을 다 통과 못 할 것 — 통과율이
    # pass_rate(기본 90%)보다 훨씬 낮은 0%로 나와야 트리거.
    graphs_root = _build_graphs(tmp_path)
    result = check_drift(
        ZeroModel(), None,
        monitor_graphs_root=graphs_root, monitor_raw_dir=FIXTURES, monitor_run_ids=RUN_IDS,
    )
    assert result["triggered"] is True
    assert result["n"] == len(RUN_IDS)
    assert result["pass_rate"] < result["required_pass_rate"]


def test_check_drift_does_not_trigger_when_pass_rate_requirement_is_zero(
    tmp_path: pathlib.Path,
) -> None:
    # pass_rate=0.0이면 무조건 통과(0% 이상은 항상 참) — 임계치 자체를 느슨하게 바꾸면
    # 트리거가 안 걸리는 것도 확인(회귀 방지용 경계 테스트).
    graphs_root = _build_graphs(tmp_path)
    result = check_drift(
        ZeroModel(), None,
        monitor_graphs_root=graphs_root, monitor_raw_dir=FIXTURES, monitor_run_ids=RUN_IDS,
        pass_rate=0.0,
    )
    assert result["triggered"] is False
