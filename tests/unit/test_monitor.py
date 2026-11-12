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


def test_check_drift_triggers_when_mean_chamfer_exceeds_tiny_threshold(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = _build_graphs(tmp_path)
    result = check_drift(
        ZeroModel(), None, champion_golden_chamfer=1e-6,
        monitor_graphs_root=graphs_root, monitor_raw_dir=FIXTURES, monitor_run_ids=RUN_IDS,
    )
    assert result["triggered"] is True
    assert result["drift_threshold"] == 1.5e-6
    assert result["mean_chamfer"] > result["drift_threshold"]


def test_check_drift_does_not_trigger_when_threshold_is_huge(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = _build_graphs(tmp_path)
    result = check_drift(
        ZeroModel(), None, champion_golden_chamfer=1e6,
        monitor_graphs_root=graphs_root, monitor_raw_dir=FIXTURES, monitor_run_ids=RUN_IDS,
    )
    assert result["triggered"] is False
