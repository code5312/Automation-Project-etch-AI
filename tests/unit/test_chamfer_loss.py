"""W4-02: 미분 가능 Chamfer 손실 단위 테스트. eval/metrics.py의 chamfer_distance(평가용,
비미분)와 같은 값을 내야 하고, 역전파가 실제로 되는지 확인한다."""

import numpy as np
import torch

from etchpipe.eval.metrics import chamfer_distance
from etchpipe.train.chamfer_loss import chamfer_loss


def test_matches_numpy_chamfer_distance_on_random_points() -> None:
    rng = np.random.default_rng(0)
    pred_np = rng.normal(size=(12, 2)).astype(np.float32)
    true_np = rng.normal(size=(9, 2)).astype(np.float32)

    expected = chamfer_distance(pred_np, true_np)
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
