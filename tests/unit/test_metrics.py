"""W2-06: Chamfer·depth 지표 단위 테스트. 설계서 §9: 해석적으로 답을 아는 입력으로 검증."""

import numpy as np

from etchpipe.eval.metrics import chamfer_distance, depth, depth_error


def test_chamfer_zero_when_identical() -> None:
    p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    assert chamfer_distance(p, p) == 0.0


def test_chamfer_order_invariant() -> None:
    p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, -1.0]])
    q = p[::-1].copy()
    a = chamfer_distance(p, q)
    b = chamfer_distance(q, p)
    assert abs(a - b) < 1e-9


def test_chamfer_parallel_lines_known_distance() -> None:
    # 두 평행한 직선(y=0, y=d)을 조밀하게 샘플링하면 양방향 최근접 거리는 모두 d에 수렴
    x = np.linspace(0, 10, 200)
    line_a = np.stack([x, np.zeros_like(x)], axis=1)
    d = 2.5
    line_b = np.stack([x, np.full_like(x, d)], axis=1)
    cd = chamfer_distance(line_a, line_b)
    assert abs(cd - 2 * d) < 0.05  # 양방향 합이므로 2*d


def test_depth_on_rectangular_trench() -> None:
    # 사각 트렌치: 위쪽(y=0) 평평한 면 + 바닥(y=-depth) 평평한 면
    top = np.stack([np.linspace(-5, 5, 20), np.zeros(20)], axis=1)
    bottom = np.stack([np.linspace(-2, 2, 10), np.full(10, -7.0)], axis=1)
    points = np.concatenate([top, bottom])
    assert depth(points) == 7.0


def test_depth_error_zero_when_identical() -> None:
    p = np.array([[0.0, 0.0], [1.0, -3.0]])
    assert depth_error(p, p) == 0.0


def test_depth_error_detects_difference() -> None:
    pred = np.array([[0.0, 0.0], [1.0, -3.0]])
    true = np.array([[0.0, 0.0], [1.0, -5.0]])
    assert depth_error(pred, true) == 2.0
