"""W2-05: 더미 모델 2종 단위 테스트."""

import numpy as np

from etchpipe.models.dummy import RandomDisplacementModel, ZeroDisplacementModel


def test_zero_displacement_returns_same_positions() -> None:
    pos = np.array([[1.0, 2.0], [3.0, -4.0]])
    out = ZeroDisplacementModel().predict(pos)
    np.testing.assert_array_equal(out, pos)


def test_zero_displacement_does_not_mutate_input() -> None:
    pos = np.array([[1.0, 2.0]])
    out = ZeroDisplacementModel().predict(pos)
    out[0, 0] = 999.0
    assert pos[0, 0] == 1.0


def test_random_displacement_changes_positions() -> None:
    pos = np.zeros((10, 2))
    out = RandomDisplacementModel(scale=1.0, seed=0).predict(pos)
    assert not np.allclose(out, pos)


def test_random_displacement_same_seed_same_result() -> None:
    pos = np.zeros((5, 2))
    a = RandomDisplacementModel(scale=1.0, seed=42).predict(pos)
    b = RandomDisplacementModel(scale=1.0, seed=42).predict(pos)
    np.testing.assert_array_equal(a, b)


def test_random_displacement_different_seed_different_result() -> None:
    pos = np.zeros((5, 2))
    a = RandomDisplacementModel(scale=1.0, seed=1).predict(pos)
    b = RandomDisplacementModel(scale=1.0, seed=2).predict(pos)
    assert not np.allclose(a, b)
