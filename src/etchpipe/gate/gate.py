"""C5: 회귀 게이트 최소판(W2-07) — 설계서 §3 C5 규칙 중 ①(골든 Chamfer)·④(롤아웃 후반 급증)만.
나머지(②이전모델대비,③추론시간,⑤)는 5주차(W5-05)에서 채운다.

종료 코드: 0=PASS, 1=BLOCK(성능 미달), 2=BLOCK(데이터 무결성 — 해시/골든셋 불일치).
"""

from dataclasses import dataclass, field

from etchpipe.eval.metrics import chamfer_distance


@dataclass
class GateResult:
    passed: bool
    exit_code: int
    reasons: list[str] = field(default_factory=list)


def check_golden_chamfer(
    model, golden_pairs: list[tuple], max_chamfer: float
) -> tuple[bool, float]:
    """항목①: 골든셋 평균 Chamfer가 임계값 이하인지. golden_pairs는 (입력 pos, 정답 pos) 목록."""
    errors = [chamfer_distance(model.predict(pred_in), true) for pred_in, true in golden_pairs]
    mean_err = sum(errors) / len(errors)
    return mean_err <= max_chamfer, mean_err


def check_rollout_late_step(step_errors: list[float], max_ratio: float = 2.0) -> bool:
    """항목④: 롤아웃 후반 절반 평균 오차가 전반 절반 평균 오차의 max_ratio배를 넘지 않는지."""
    if len(step_errors) < 2:
        return True
    half = len(step_errors) // 2
    early, late = step_errors[:half], step_errors[half:]
    return (sum(late) / len(late)) <= (sum(early) / len(early)) * max_ratio


def run_gate(
    model,
    golden_pairs: list[tuple],
    max_chamfer: float,
    rollout_step_errors: list[float] | None = None,
    rollout_max_ratio: float = 2.0,
    expected_dataset_hash: str | None = None,
    actual_dataset_hash: str | None = None,
    expected_golden_hash: str | None = None,
    actual_golden_hash: str | None = None,
) -> GateResult:
    """데이터 무결성(종료코드 2)은 두 가지를 독립적으로 본다: 학습에 쓰인 dataset_hash가
    기대값과 같은지(설계서 §4.4), 골든셋 자체가 변조되지 않았는지."""
    if expected_dataset_hash is not None and actual_dataset_hash != expected_dataset_hash:
        msg = f"dataset_hash mismatch: expected {expected_dataset_hash}, got {actual_dataset_hash}"
        return GateResult(passed=False, exit_code=2, reasons=[msg])
    if expected_golden_hash is not None and actual_golden_hash != expected_golden_hash:
        msg = f"golden hash mismatch: expected {expected_golden_hash}, got {actual_golden_hash}"
        return GateResult(passed=False, exit_code=2, reasons=[msg])

    reasons = []
    ok1, mean_chamfer = check_golden_chamfer(model, golden_pairs, max_chamfer)
    if not ok1:
        reasons.append(f"golden chamfer {mean_chamfer:.4f} > max {max_chamfer:.4f}")

    ok4 = True
    if rollout_step_errors is not None:
        ok4 = check_rollout_late_step(rollout_step_errors, rollout_max_ratio)
        if not ok4:
            reasons.append("rollout late-step error spike")

    passed = ok1 and ok4
    return GateResult(passed=passed, exit_code=0 if passed else 1, reasons=reasons)
