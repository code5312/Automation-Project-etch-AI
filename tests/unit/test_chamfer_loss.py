"""W4-02: 미분 가능 Chamfer 손실 단위 테스트. ADR-0004 이후로는 eval/metrics.py의
chamfer_distance(점-선분, densify, ÷2 — 평가·게이트용)와 더 이상 같은 정의가 아니다
(의도적으로 점-점·합 방식 유지, 역전파 속도 때문). 그래서 비교 대상은 eval 쪽 함수가
아니라, 같은 점-점 정의를 scipy로 독립 구현한 참조값이다. 역전파가 실제로 되는지도
확인한다."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from etchpipe.train.chamfer_loss import chamfer_loss


def _reference_point_point_chamfer(pred: np.ndarray, true: np.ndarray) -> float:
    """chamfer_loss와 같은 정의(점-점 최근접, 양방향 합)의 독립 참조 구현."""
    d_pred_to_true, _ = cKDTree(true).query(pred)
    d_true_to_pred, _ = cKDTree(pred).query(true)
    return float(d_pred_to_true.mean() + d_true_to_pred.mean())


def test_matches_point_point_reference_on_random_points() -> None:
    rng = np.random.default_rng(0)
    pred_np = rng.normal(size=(12, 2)).astype(np.float32)
    true_np = rng.normal(size=(9, 2)).astype(np.float32)

    expected = _reference_point_point_chamfer(pred_np, true_np)
    actual = chamfer_loss(torch.tensor(pred_np), torch.tensor(true_np)).item()
    assert abs(expected - actual) < 1e-5


def test_zero_when_point_sets_identical() -> None:
    pts = torch.rand(10, 2)
    assert chamfer_loss(pts, pts).item() < 1e-6


def test_gradient_flows_to_pred() -> None:
    pred = torch.rand(8, 2, requires_grad=True)
    true = torch.rand(6, 2)
    loss = chamfer_loss(pred, true)
    loss.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)


def test_loss_decreases_as_pred_moves_toward_true() -> None:
    true = torch.tensor([[10.0, 10.0]] * 5)
    far = torch.zeros(5, 2)
    near = torch.full((5, 2), 9.0)
    assert chamfer_loss(near, true).item() < chamfer_loss(far, true).item()
