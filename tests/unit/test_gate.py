"""W2-07: 게이트 최소판(항목 ①·④, 종료코드 0/1/2) 단위 테스트."""

import numpy as np

from etchpipe.gate.gate import check_rollout_late_step, run_gate
from etchpipe.models.dummy import RandomDisplacementModel, ZeroDisplacementModel

_LINE = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
_STEP = np.array([[0.0, -1.0], [1.0, -1.0]])
GOLDEN_PAIRS = [(_LINE, _LINE), (_STEP, _STEP)]


def test_zero_model_passes_exit_code_0() -> None:
    result = run_gate(ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1)
    assert result.passed is True
    assert result.exit_code == 0


def test_random_model_blocks_exit_code_1() -> None:
    result = run_gate(RandomDisplacementModel(scale=5.0, seed=0), GOLDEN_PAIRS, max_chamfer=0.1)
    assert result.passed is False
    assert result.exit_code == 1
    assert result.reasons


def test_dataset_hash_mismatch_blocks_exit_code_2() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        expected_dataset_hash="abc123", actual_dataset_hash="different",
    )
    assert result.passed is False
    assert result.exit_code == 2


def test_dataset_hash_match_does_not_block() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        expected_dataset_hash="abc123", actual_dataset_hash="abc123",
    )
    assert result.exit_code == 0


def test_golden_hash_mismatch_blocks_exit_code_2() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        expected_golden_hash="abc123", actual_golden_hash="different",
    )
    assert result.passed is False
    assert result.exit_code == 2


def test_golden_hash_match_does_not_block() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        expected_golden_hash="abc123", actual_golden_hash="abc123",
    )
    assert result.exit_code == 0


def test_rollout_late_step_no_spike_passes() -> None:
    assert check_rollout_late_step([0.1, 0.1, 0.11, 0.12]) is True


def test_rollout_late_step_spike_fails() -> None:
    assert check_rollout_late_step([0.1, 0.1, 5.0, 8.0], max_ratio=2.0) is False
