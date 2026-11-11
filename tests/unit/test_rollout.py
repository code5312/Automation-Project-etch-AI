"""W4-05: 롤아웃 v1(연결 유지, 법선·엣지 재계산, 재료 이전 값 유지) 단위 테스트."""

import pathlib

import numpy as np
import torch
from torch import nn

from etchpipe.etl.etl import THETA_KEYS, etl_run, load_snapshot, outer_path, to_graph
from etchpipe.eval.metrics import depth
from etchpipe.train.loop import load_graph
from etchpipe.train.rollout import (
    lines_from_edge_index,
    recompute_edge_features,
    recompute_normals,
    rollout,
    rollout_val_error,
    rollout_val_error_by_step,
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


class ConstantDisplacementModel(nn.Module):
    """항상 같은 변위를 모든 노드에 반환 — fix_mask/guard 검증용."""

    def __init__(self, dx: float = 0.0, dy: float = 0.0) -> None:
        super().__init__()
        self.dx, self.dy = dx, dy

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return torch.tensor([[self.dx, self.dy]] * node.shape[0])


def test_rollout_fix_mask_keeps_mask_nodes_in_place() -> None:
    # ADR-0004 후속(2026-11-11): mgn_tuned inference.py의 마스크 고정 포팅.
    pos0, edge_index = _line_geometry()
    normal0 = torch.tensor([[0.0, 1.0]] * 3)
    material0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # 점0=마스크, 1·2=Si
    theta0 = torch.zeros((3, 7))
    t0 = torch.zeros((3, 1))
    node0 = torch.cat([pos0, normal0, material0, theta0, t0], dim=1)
    edge0 = recompute_edge_features(pos0, edge_index)

    positions = rollout(
        ConstantDisplacementModel(dx=1.0), node0, edge_index, edge0, pos0, 2, 1.0, fix_mask=True
    )
    assert torch.allclose(positions[-1][0], pos0[0])  # 마스크 노드는 안 움직임
    assert torch.allclose(positions[-1][1], pos0[1] + torch.tensor([2.0, 0.0]))  # Si는 그대로 누적


def _cross_geometry() -> tuple[torch.Tensor, torch.Tensor]:
    # 선분 A(0-1, y=1 수평) + 선분 B(2-3, x=0, y=-1~0.5, A와 아직 안 교차) — 두 선분은
    # 서로 연결 안 됨(독립), 교차가드는 연결 여부와 무관하게 선분-선분 교차만 본다.
    pos = torch.tensor([[-1.0, 1.0], [1.0, 1.0], [0.0, -1.0], [0.0, 0.5]])
    edge_index = torch.tensor([[0, 2, 1, 3], [1, 3, 0, 2]], dtype=torch.int64)
    return pos, edge_index


def test_rollout_guard_shrinks_displacement_that_would_newly_cross() -> None:
    pos0, edge_index = _cross_geometry()
    normal0 = torch.tensor([[0.0, 1.0]] * 4)
    material0 = torch.tensor([[0.0, 1.0]] * 4)
    theta0 = torch.zeros((4, 7))
    t0 = torch.zeros((4, 1))
    node0 = torch.cat([pos0, normal0, material0, theta0, t0], dim=1)
    edge0 = recompute_edge_features(pos0, edge_index)

    class MovePoint3(nn.Module):
        def forward(self, node, edge_index, edge):  # noqa: ANN001
            del edge_index, edge
            d = torch.zeros((node.shape[0], 2))
            d[3] = torch.tensor([0.0, 1.0])  # 점3을 위로 1.0 이동 -> 선분A와 새로 교차
            return d

    without_guard = rollout(MovePoint3(), node0, edge_index, edge0, pos0, 1, 1.0, guard=False)
    assert without_guard[-1][3, 1] > 1.0  # 교차 생김(y=1.5, 선분A를 완전히 지나침)

    with_guard = rollout(MovePoint3(), node0, edge_index, edge0, pos0, 1, 1.0, guard=True)
    assert with_guard[-1][3, 1] <= 1.0 + 1e-5  # 가드가 교차 직전까지만 허용(접촉, 관통 안 함)


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

    assert set(result) == {"chamfer", "depth_error", "depth_pred", "depth_true"}
    assert result["chamfer"] >= 0.0
    assert result["depth_error"] >= 0.0
    # depth_pred는 음수일 수 있다(ToyNodeMLP는 미학습이라 표면이 y=0 위로 발산하기도 함,
    # metrics.depth()의 정의 -ymin 그대로) — depth_true만 실제 시뮬레이션 데이터라 >=0 보장.
    assert result["depth_true"] >= 0.0


def test_rollout_val_error_by_step_reports_each_requested_step(tmp_path: pathlib.Path) -> None:
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = rollout_val_error_by_step(model, graphs_root, "r0000", steps=[1, 2, 3], step_min=1.0)

    assert set(result) == {1, 2, 3}
    for s in (1, 2, 3):
        assert set(result[s]) == {"chamfer", "depth_error", "depth_pred", "depth_true"}
        assert result[s]["chamfer"] >= 0.0


def test_rollout_val_error_by_step_matches_rollout_val_error_at_same_step(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    torch.manual_seed(0)
    model = ToyNodeMLP()
    single = rollout_val_error(model, graphs_root, "r0000", n_steps=3, step_min=1.0)
    torch.manual_seed(0)
    model = ToyNodeMLP()
    by_step = rollout_val_error_by_step(model, graphs_root, "r0000", steps=[3], step_min=1.0)

    assert single == by_step[3]


def test_lines_from_edge_index_recovers_original_lines() -> None:
    # ADR-0004 물리 규칙 배선: to_graph의 실제 양방향 엣지 구성(앞 절반=원본 선분)을
    # 그대로 역산하는지 real fixture로 확인.
    p, lines, mid, nv = load_snapshot(FIXTURES / "r0000" / "snapshots" / "t000.vtp")
    theta = np.zeros(7, dtype=np.float32)
    _, ei, _ = to_graph(p, lines, mid, nv, theta, t=0.0)
    recovered = lines_from_edge_index(torch.from_numpy(ei))
    np.testing.assert_array_equal(recovered, lines)


def test_rollout_val_error_by_step_without_physics_check_omits_self_intersections(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = rollout_val_error_by_step(model, graphs_root, "r0000", steps=[1], step_min=1.0)
    assert "self_intersections" not in result[1]


def test_rollout_val_error_by_step_with_physics_check_counts_zero_for_unmoved_surface(
    tmp_path: pathlib.Path,
) -> None:
    # 변위 0(ZeroDisplacementModel류) -> 원본 표면 그대로 -> 자기교차 없음(0).
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    model = RecordingZeroModel()
    result = rollout_val_error_by_step(
        model, graphs_root, "r0000", steps=[1], step_min=1.0, check_physics=True
    )
    assert result[1]["self_intersections"] == 0


def test_rollout_val_error_by_step_reports_depth_pred_true_raw_values(
    tmp_path: pathlib.Path,
) -> None:
    # W5-06 배선: 변위 0 모델이면 depth_pred는 t=0 표면 깊이 그대로여야 하고,
    # depth_error는 depth_pred/depth_true의 절대차와 같아야 한다.
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    model = RecordingZeroModel()
    result = rollout_val_error_by_step(model, graphs_root, "r0000", steps=[1], step_min=1.0)

    g0 = load_graph(graphs_root / "r0000" / "g000.npz")
    pred_lines = lines_from_edge_index(g0["edge_index"])
    expected_depth0 = depth(outer_path(g0["pos"].numpy(), pred_lines))

    assert abs(result[1]["depth_pred"] - expected_depth0) < 1e-6
    raw_diff = abs(result[1]["depth_pred"] - result[1]["depth_true"])
    assert abs(result[1]["depth_error"] - raw_diff) < 1e-6


def test_rollout_val_error_by_step_with_physics_check_falls_back_on_divergence(
    tmp_path: pathlib.Path,
) -> None:
    # ToyNodeMLP(미학습, norm 없이 실제 θ 크기 그대로 입력)는 1스텝만에 도메인 밖으로
    # 발산해 outer_path가 왼쪽/오른쪽 끝점을 못 찾는다 — 예외 대신 self_intersections=1
    # (명백한 물리 위반)로 떨어져야 한다.
    graphs_root = tmp_path / "graphs"
    etl_run(FIXTURES / "r0000", graphs_root / "r0000", THETA_KEYS)

    torch.manual_seed(0)
    model = ToyNodeMLP()
    result = rollout_val_error_by_step(
        model, graphs_root, "r0000", steps=[1], step_min=1.0, check_physics=True
    )
    assert result[1]["self_intersections"] == 1
