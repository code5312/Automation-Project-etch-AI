"""W4-05: 롤아웃 v1(연결 유지, 법선·엣지 재계산, 재료 이전 값 유지) 단위 테스트."""

import pathlib

import torch
from torch import nn

from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.train.rollout import (
    recompute_edge_features,
    recompute_normals,
    rollout,
    rollout_val_error,
)

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _line_geometry() -> tuple[torch.Tensor, torch.Tensor]:
    # 수평선 3점: 0-1-2, 양방향 엣지(etl.to_graph와 같은 구성)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    return pos, edge_index


def test_recompute_edge_features_matches_definition() -> None:
    pos, edge_index = _line_geometry()
    edge = recompute_edge_features(pos, edge_index)
    assert edge.shape == (4, 3)
    assert torch.allclose(edge[0], torch.tensor([1.0, 0.0, 1.0]))  # 0->1: rel=(1,0), dist=1


def test_recompute_normals_unit_length_and_matches_hand_calc() -> None:
    pos, edge_index = _line_geometry()
    prev_normal = torch.tensor([[0.0, 1.0]] * 3)
    normal = recompute_normals(pos, edge_index, prev_normal)
    assert normal.shape == (3, 2)
    assert torch.allclose(normal, torch.tensor([[0.0, 1.0]] * 3), atol=1e-6)
    assert torch.allclose(normal.norm(dim=1), torch.ones(3), atol=1e-6)


def test_recompute_normals_sign_continuity_with_flipped_prev() -> None:
    pos, edge_index = _line_geometry()
    prev_normal = torch.tensor([[0.0, -1.0]] * 3)
    normal = recompute_normals(pos, edge_index, prev_normal)
    assert torch.allclose(normal, torch.tensor([[0.0, -1.0]] * 3), atol=1e-6)


class RecordingZeroModel(nn.Module):
    """변위 항상 0 — 연결·재료·theta·t가 스텝마다 어떻게 바뀌는지 기록해서 검증하는 용도."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((node.clone(), edge_index.clone(), edge.clone()))
        return torch.zeros((node.shape[0], 2))


def test_rollout_keeps_connectivity_material_theta_and_increments_t() -> None:
    pos0, edge_index = _line_geometry()
    normal0 = torch.tensor([[0.0, 1.0]] * 3)
    material0 = torch.tensor([[0.0, 1.0]] * 3)  # Si
    theta0 = torch.arange(7, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
    t0 = torch.zeros((3, 1))
    node0 = torch.cat([pos0, normal0, material0, theta0, t0], dim=1)
    edge0 = recompute_edge_features(pos0, edge_index)

    model = RecordingZeroModel()
    step_min = 1.5
    n_steps = 4
    positions = rollout(model, node0, edge_index, edge0, pos0, n_steps, step_min)

    assert len(positions) == n_steps + 1
    assert len(model.calls) == n_steps
    for i, (node, ei, _edge) in enumerate(model.calls):
        assert torch.equal(ei, edge_index)  # 연결 유지
        assert torch.allclose(node[:, 4:6], material0)  # 재료 유지
        assert torch.allclose(node[:, 6:13], theta0)  # theta 유지
        expected_t = i * step_min
        assert torch.allclose(node[:, 13], torch.full((3,), expected_t))
    # 변위가 항상 0이므로 위치는 그대로
    assert torch.allclose(positions[-1], pos0)


class MaterialEchoModel(nn.Module):
    """변위 = material 채널 그대로 — 재료가 매 스텝 제대로 들고 가는지 위치 누적으로 검증."""

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return node[:, 4:6]


def test_rollout_material_carried_forward_via_cumulative_displacement() -> None:
    pos0, edge_index = _line_geometry()
    normal0 = torch.tensor([[0.0, 1.0]] * 3)
    material0 = torch.tensor([[0.2, 0.5]] * 3)
    theta0 = torch.zeros((3, 7))
    t0 = torch.zeros((3, 1))
    node0 = torch.cat([pos0, normal0, material0, theta0, t0], dim=1)
    edge0 = recompute_edge_features(pos0, edge_index)

    n_steps = 3
    positions = rollout(MaterialEchoModel(), node0, edge_index, edge0, pos0, n_steps, 1.0)

    expected_final = pos0 + n_steps * material0
    assert torch.allclose(positions[-1], expected_final, atol=1e-5)


class ToyNodeMLP(nn.Module):
    def __init__(self, n_in: int = 14, n_out: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return self.lin(node)


def test_rollout_val_error_runs_on_real_fixture_graphs(tmp_path: pathlib.Path) -> None:
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = rollout_val_error(model, graphs_root, "r0000", n_steps=3, step_min=1.0)

    assert set(result) == {"chamfer", "depth_error"}
    assert result["chamfer"] >= 0.0
    assert result["depth_error"] >= 0.0
