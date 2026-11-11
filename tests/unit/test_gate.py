"""W2-07/W5-05/ADR-0004: 게이트(항목 ①~⑤+물리, 챔피언 없음 케이스, 종료코드 0/1/2,
gate_result.json) 단위 테스트. ADR-0004 이후 ①③은 참고 지표로만 남고 PASS/BLOCK에는
②④+물리만 영향을 준다."""

import json
from pathlib import Path

import numpy as np

from etchpipe.gate.gate import (
    check_cd_depth_error,
    check_champion_regression,
    check_dimension_error,
    check_latency,
    check_rollout_late_step,
    check_rollout_late_step_multi,
    run_gate,
    write_gate_result,
)
from etchpipe.models.dummy import RandomDisplacementModel, ZeroDisplacementModel

_LINE = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
_STEP = np.array([[0.0, -1.0], [1.0, -1.0]])
GOLDEN_PAIRS = [(_LINE, _LINE), (_STEP, _STEP)]


def test_zero_model_passes_exit_code_0() -> None:
    result = run_gate(ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1)
    assert result.passed is True
    assert result.exit_code == 0


def test_random_model_does_not_block_alone_since_rule1_is_reference_only() -> None:
    # ADR-0004 결정2: ①(골든 Chamfer 상한)은 더 이상 차단하지 않는다. ②③④물리를 전부
    # 비워 둔 호출에서는 ①만 깨져도 BLOCK이 아니라 PASS가 나와야 한다.
    result = run_gate(RandomDisplacementModel(scale=5.0, seed=0), GOLDEN_PAIRS, max_chamfer=0.1)
    assert result.passed is True
    assert result.exit_code == 0
    assert result.reference_metrics["golden_chamfer"]["within_max"] is False


def test_empty_golden_pairs_skips_rule1_without_crashing() -> None:
    # W5-06 2026-11-12: 실모델(GNN rollout) gate-negative 테스트는 더미 모델 전용
    # `model.predict(pos)` 인터페이스에 맞는 골든 어댑터가 없어 golden_pairs=[]로
    # run_gate를 호출한다 — ①은 참고 지표일 뿐이라 건너뛰어도 passed에 영향 없어야 한다.
    result = run_gate(ZeroDisplacementModel(), [], max_chamfer=0.1)
    assert result.passed is True
    assert "golden_chamfer" not in result.reference_metrics


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


def test_rollout_late_step_multi_averages_by_step_index_across_recipes() -> None:
    # ADR-0004 결정3: 레시피 2개를 스텝 인덱스 기준으로 먼저 평균한 뒤 반/반 비율.
    # step0: (0.1+0.3)/2=0.2, step1: (0.1+0.3)/2=0.2, step2: 1.0(레시피1은 스텝 2가
    # 없어 빠짐) → chamfer_by_step=[0.2, 0.2, 1.0] → 전반[0.2] 후반[0.2,1.0]=0.6 →
    # 비율 3.0 > 2.0 → BLOCK.
    ok, ratio = check_rollout_late_step_multi([[0.1, 0.1], [0.3, 0.3, 1.0]], max_ratio=2.0)
    assert ok is False
    assert ratio > 2.0


def test_rollout_late_step_multi_near_zero_errors_treated_as_ratio_one() -> None:
    ok, ratio = check_rollout_late_step_multi([[1e-9, 1e-9, 1e-9, 1e-9]])
    assert ok is True
    assert ratio == 1.0


def test_check_dimension_error_uses_percent_when_true_value_large() -> None:
    # grid=0.3, max_pct=5 → 경계값 6.0µm. true=10(≥6) → %오차 기준.
    ok, detail = check_dimension_error([10.4], [10.0], grid=0.3, max_pct=5.0)
    assert ok is True
    assert detail["n_large"] == 1 and detail["n_small"] == 0


def test_check_dimension_error_uses_absolute_when_true_value_small() -> None:
    # true=2(<6) → 절대오차(≤grid=0.3) 기준. |2.2-2.0|=0.2 ≤ 0.3 → 통과.
    ok, detail = check_dimension_error([2.2], [2.0], grid=0.3, max_pct=5.0)
    assert ok is True
    assert detail["n_small"] == 1 and detail["n_large"] == 0


def test_check_dimension_error_small_value_fails_when_absolute_error_too_big() -> None:
    # |2.5-2.0|=0.5 > grid=0.3 → 불합격 (같은 %였으면 25%라 통과처럼 보일 수 있었음).
    ok, _ = check_dimension_error([2.5], [2.0], grid=0.3, max_pct=5.0)
    assert ok is False


def test_check_cd_depth_error_combines_depth_and_cd() -> None:
    ok, detail = check_cd_depth_error(
        depth_pred=[10.4], depth_true=[10.0], cd_pred=[20.0], cd_true=[20.0], grid=0.3, max_pct=5.0
    )
    assert ok is True
    assert "depth" in detail and "cd" in detail


def test_check_cd_depth_error_fails_if_either_side_fails() -> None:
    ok, _ = check_cd_depth_error(
        depth_pred=[10.0], depth_true=[10.0], cd_pred=[25.0], cd_true=[20.0], grid=0.3, max_pct=5.0
    )
    assert ok is False


def test_check_champion_regression_true_when_no_champion() -> None:
    # 체크리스트 W5-05 "챔피언 없음 케이스": 비교 대상이 없으면 자동 통과
    assert check_champion_regression(current_mean_error=999.0, champion_mean_error=None) is True


def test_check_champion_regression_passes_within_bound() -> None:
    assert check_champion_regression(1.05, champion_mean_error=1.0, max_regression=0.10) is True


def test_check_champion_regression_fails_beyond_bound() -> None:
    assert check_champion_regression(1.2, champion_mean_error=1.0, max_regression=0.10) is False


def test_check_latency_none_when_not_measured() -> None:
    assert check_latency(None, 1.0) is None
    assert check_latency(0.5, None) is None


def test_check_latency_warns_without_blocking() -> None:
    assert check_latency(2.0, max_latency_s=1.0) is not None
    assert check_latency(0.5, max_latency_s=1.0) is None


def test_run_gate_blocks_on_cd_depth_error_over_threshold() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        depth_pred=[20.0], depth_true=[10.0], cd_pred=[20.0], cd_true=[20.0],
        grid=0.3, max_dim_pct=5.0,
    )
    assert result.passed is False
    assert result.exit_code == 1


def test_run_gate_does_not_block_on_champion_regression_reference_only() -> None:
    # ADR-0004 결정2: ③은 더 이상 차단하지 않는다 — reference_metrics에만 남는다.
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        current_mean_error=2.0, champion_mean_error=1.0, max_regression=0.10,
    )
    assert result.passed is True
    assert result.reference_metrics["champion_regression"]["within_max"] is False


def test_run_gate_passes_when_no_champion_yet() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        current_mean_error=999.0, champion_mean_error=None,
    )
    assert result.passed is True


def test_run_gate_blocks_on_rollout_multi_recipe_ratio() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        rollout_step_errors_by_recipe=[[0.1, 0.1, 5.0, 8.0]],
    )
    assert result.passed is False
    assert result.exit_code == 1


def test_run_gate_blocks_on_self_intersection() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1, self_intersection_count=2,
    )
    assert result.passed is False
    assert result.reference_metrics["physics"]["self_intersection_count"] == 2


def test_run_gate_blocks_on_model_specific_violations() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1, model_specific_violations=3,
    )
    assert result.passed is False


def test_run_gate_skips_physics_when_unmeasured() -> None:
    # 사용자 결정 2026-11-11: model_specific_violations=None(미확인)이면 그 부분은
    # 기존 관례(②③⑤와 동일)대로 건너뛴다 — 차단하지 않는다.
    result = run_gate(ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1)
    assert result.passed is True
    assert "physics" not in result.reference_metrics


def test_run_gate_latency_over_budget_warns_but_still_passes() -> None:
    result = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1,
        latency_s=5.0, max_latency_s=1.0,
    )
    assert result.passed is True
    assert result.exit_code == 0
    assert result.warnings


def test_write_gate_result_writes_json_with_thresholds(tmp_path: Path) -> None:
    result = run_gate(ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=0.1)
    out_path = tmp_path / "gate_result.json"

    write_gate_result(result, out_path, thresholds={"max_chamfer": 0.1})

    saved = json.loads(out_path.read_text())
    assert saved["passed"] is True
    assert saved["exit_code"] == 0
    assert saved["thresholds"] == {"max_chamfer": 0.1}
