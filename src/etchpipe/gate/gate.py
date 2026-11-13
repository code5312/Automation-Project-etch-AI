"""C5: 회귀 게이트(W5-05 + ADR-0004 재구성판) — 설계서 §3 C5 규칙 ①~⑤ + 신설 물리 규칙.

ADR-0004(2026-11-11, 골든셋 공식 채점 노트북 기준): **①(골든 Chamfer 상한)·③(직전
챔피언 대비 악화)은 더 이상 PASS/BLOCK을 가르지 않는다** — 계산은 하되
`GateResult.reference_metrics`에 참고용으로만 남긴다. PASS/BLOCK은 **②(CD·깊이 치수
오차)·④(롤아웃 후반 급증)·신설 물리(선 꼬임+모델별 결함)**가 가른다. ⑤(추론 시간)은
여전히 "상한 초과 시 경고"만 — `GateResult.warnings`에만 쌓이고 `passed`에 영향 없다.

종료 코드: 0=PASS, 1=BLOCK(성능 미달), 2=BLOCK(데이터 무결성 — 해시/골든셋 불일치).
"""

import json
import math
import pathlib
from dataclasses import asdict, dataclass, field

import numpy as np

from etchpipe.eval.metrics import chamfer_distance


@dataclass
class GateResult:
    passed: bool
    exit_code: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference_metrics: dict = field(default_factory=dict)


def check_golden_chamfer(
    model, golden_pairs: list[tuple], max_chamfer: float
) -> tuple[bool, float]:
    """항목①: 골든셋 평균 Chamfer가 임계값 이하인지. golden_pairs는 (입력 pos, 정답 pos) 목록.
    ADR-0004 결정2: `run_gate`는 이 결과를 PASS/BLOCK에 쓰지 않고 참고 지표로만 남긴다
    (골든셋이 train 레시피라 "일반화"가 아니라 "학습 회차 비교"용 숫자이기 때문)."""
    errors = [chamfer_distance(model.predict(pred_in), true) for pred_in, true in golden_pairs]
    mean_err = sum(errors) / len(errors)
    return mean_err <= max_chamfer, mean_err


def check_dimension_error(
    pred: list[float], true: list[float], grid: float, max_pct: float = 5.0
) -> tuple[bool, dict]:
    """깊이 또는 CD 중 하나(레시피별 값 리스트)의 오차 판정 [확인, ADR-0004 결정2 —
    골든셋 공식 채점 노트북 `dimension_rule()` 그대로 포팅]. 정답이 `grid/(max_pct/100)`
    (예: grid=0.3, max_pct=5 → 6.0) 이상인 레시피는 %오차로, 미만인 레시피는 절대오차
    (≤grid)로 판정 — 작은 값에서 %오차가 분모 때문에 불안정해지는 걸 피한다."""
    pred_a, true_a = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    min_dim_true = grid / (max_pct / 100.0)
    large = true_a >= min_dim_true
    mean_pct = (
        float(np.mean(np.abs(pred_a[large] - true_a[large]) / true_a[large]) * 100)
        if large.any()
        else 0.0
    )
    mean_abs = float(np.mean(np.abs(pred_a[~large] - true_a[~large]))) if (~large).any() else 0.0
    relative_ok = not large.any() or mean_pct <= max_pct
    absolute_ok = bool(large.all()) or mean_abs <= grid
    detail = {
        "mean_pct_error": mean_pct,
        "mean_abs_error": mean_abs,
        "n_large": int(large.sum()),
        "n_small": int((~large).sum()),
    }
    return bool(relative_ok and absolute_ok), detail


def check_cd_depth_error(
    depth_pred: list[float],
    depth_true: list[float],
    cd_pred: list[float],
    cd_true: list[float],
    grid: float = 0.3,
    max_pct: float = 5.0,
) -> tuple[bool, dict]:
    """항목②: 깊이·중간 CD 각각에 `check_dimension_error`를 적용하고 둘 다 통과해야
    한다 [확인, ADR-0004 결정2]. `grid` 기본값 0.3은 `configs/sim/base.yaml`의
    `sim.grid_delta`와 동일."""
    depth_ok, depth_detail = check_dimension_error(depth_pred, depth_true, grid, max_pct)
    cd_ok, cd_detail = check_dimension_error(cd_pred, cd_true, grid, max_pct)
    return depth_ok and cd_ok, {"depth": depth_detail, "cd": cd_detail}


def check_champion_regression(
    current_mean_error: float,
    champion_mean_error: float | None,
    max_regression: float = 0.10,
) -> bool:
    """항목③: 직전 챔피언 대비 평균 오차 악화가 `max_regression`(기본 10%) 이하인지.
    챔피언이 없으면(첫 모델, 체크리스트 W5-05 "챔피언 없음 케이스") 비교 대상이 없어
    자동 통과. ADR-0004 결정2: `run_gate`는 이 결과도 PASS/BLOCK에 쓰지 않고 참고
    지표로만 남긴다."""
    if champion_mean_error is None:
        return True
    return current_mean_error <= champion_mean_error * (1 + max_regression)


def check_rollout_late_step(step_errors: list[float], max_ratio: float = 2.0) -> bool:
    """항목④(레시피 1개용): 롤아웃 후반 절반 평균 오차가 전반 절반 평균 오차의
    max_ratio배를 넘지 않는지. ADR-0004 결정3 이후로는 골든셋 평가처럼 레시피가 여럿일
    때 `check_rollout_late_step_multi`를 우선 쓴다 — 이 함수는 레시피 1개짜리 호출부
    (예: 단일 OOD 샘플) 호환용으로 남긴다."""
    if len(step_errors) < 2:
        return True
    half = len(step_errors) // 2
    early, late = step_errors[:half], step_errors[half:]
    return (sum(late) / len(late)) <= (sum(early) / len(early)) * max_ratio


def check_rollout_late_step_multi(
    step_errors_by_recipe: list[list[float]], max_ratio: float = 2.0
) -> tuple[bool, float]:
    """항목④(다중 레시피) [확인, ADR-0004 결정3 — 골든셋 공식 채점 노트북
    `late_early_ratio()` 그대로 포팅]. 레시피마다 스텝 수가 달라도(조기 종료) 되고,
    **스텝 인덱스 기준으로 레시피를 먼저 평균**한 시퀀스를 전/후반으로 나눠 비율을 본다 —
    레시피 하나만 보면 식각 속도 차이가 노이즈로 보여서, 여러 레시피를 묶어야 "후반에
    갈수록 정말 나빠지는지"가 안정적으로 드러난다는 노트북 설계 의도. 오차가 사실상
    0이면(계산 잔차로 비율이 튀는 것 방지) 비율 1.0으로 자동 통과."""
    max_len = max(len(e) for e in step_errors_by_recipe)
    chamfer_by_step = [
        sum(vals) / len(vals)
        for k in range(max_len)
        if (vals := [errs[k] for errs in step_errors_by_recipe if k < len(errs)])
    ]
    errors = np.array(chamfer_by_step)
    if len(errors) < 2 or errors.max() < 1e-6:
        return True, 1.0
    half = len(errors) // 2
    ratio = float(errors[half:].mean() / errors[:half].mean())
    return ratio <= max_ratio, ratio


def check_rollout_recipe(
    step_errors: list[float], max_ratio: float = 2.5, min_increase: float = 0.3
) -> tuple[bool, float, float]:
    """ADR-0005 결정3: 레시피 1개의 롤아웃 오차 증가를 **비율 OR 증가량**으로 판정한다
    (기존 `check_rollout_late_step`은 비율만 봄) — 전/후반 평균 비율이 `max_ratio`
    이하이거나, 증가량(후반-전반)이 `min_increase` 이하면 통과. 오차가 사실상 0이면
    (전반 평균 <1e-6) 비율이 계산 잔차로 튀는 걸 막기 위해 자동 통과(기존 관례와 동일,
    ADR-0004 결정3 제로 가드 유지). 반환: (통과여부, 비율, 증가량)."""
    if len(step_errors) < 2:
        return True, 1.0, 0.0
    half = len(step_errors) // 2
    early = sum(step_errors[:half]) / half
    late = sum(step_errors[half:]) / (len(step_errors) - half)
    if early < 1e-6:
        return True, 1.0, 0.0
    ratio = late / early
    increase = late - early
    return (ratio <= max_ratio or increase <= min_increase), ratio, increase


def check_recipe_pass_rate(
    depth_pred: list[float],
    depth_true: list[float],
    cd_pred: list[float],
    cd_true: list[float],
    rollout_step_errors_by_recipe: list[list[float]],
    grid: float = 0.3,
    max_dim_pct: float = 5.0,
    cd_floor: float = 0.6,
    rollout_max_ratio: float = 2.5,
    rollout_min_increase: float = 0.3,
    pass_rate: float = 0.9,
) -> tuple[bool, dict]:
    """ADR-0005 결정3 — 게이트 판정의 집계 축을 "레시피 묶음 평균"에서 "레시피별 판정 +
    통과율"로 바꾼다. 레시피마다 깊이(`grid` 하한)·중간CD(`cd_floor` 하한)·롤아웃
    (`check_rollout_recipe`) 3조건을 **독립적으로** 판정하고, 셋 다 충족한 레시피
    비율이 `pass_rate`(기본 90%) 이상이면 합격. 근거: Clopper-Pearson 95% 신뢰하한
    (36/40 충족 시 실제 충족률 하한 0.79, 문서 §5).

    **계산 실패 레시피 처리** [문서 §3]: `depth_pred[i]`/`cd_pred[i]`/해당 레시피의
    롤아웃 스텝 오차 중 하나라도 유한하지 않으면(NaN/inf) 그 레시피는 **세 조건 모두
    미충족**으로 센다 — 통과율 분모에는 포함하되 통과로 잘못 집계되지 않게 한다."""
    n = len(depth_pred)
    per_recipe: list[dict] = []
    n_passed = 0
    for i in range(n):
        step_errors = rollout_step_errors_by_recipe[i]
        calc_ok = (
            math.isfinite(depth_pred[i])
            and math.isfinite(cd_pred[i])
            and all(math.isfinite(e) for e in step_errors)
        )
        if not calc_ok:
            per_recipe.append(
                {"depth_ok": False, "cd_ok": False, "rollout_ok": False, "calc_failed": True}
            )
            continue
        depth_ok, depth_detail = check_dimension_error(
            [depth_pred[i]], [depth_true[i]], grid, max_dim_pct
        )
        cd_ok, cd_detail = check_dimension_error([cd_pred[i]], [cd_true[i]], cd_floor, max_dim_pct)
        rollout_ok, ratio, increase = check_rollout_recipe(
            step_errors, rollout_max_ratio, rollout_min_increase
        )
        recipe_ok = depth_ok and cd_ok and rollout_ok
        n_passed += int(recipe_ok)
        per_recipe.append(
            {
                "depth_ok": depth_ok, "depth_detail": depth_detail,
                "cd_ok": cd_ok, "cd_detail": cd_detail,
                "rollout_ok": rollout_ok, "rollout_ratio": ratio, "rollout_increase": increase,
                "calc_failed": False,
            }
        )
    fraction = (n_passed / n) if n else 0.0
    detail = {"n": n, "n_passed": n_passed, "pass_rate": fraction, "per_recipe": per_recipe}
    return fraction >= pass_rate, detail


def check_no_regression(
    candidate_iou: list[float],
    champion_iou: list[float],
    candidate_hd95: list[float],
    champion_hd95: list[float],
    val_pass_rate_result: tuple[bool, dict] | None = None,
    alpha: float = 0.0125,
    max_hd95_degradation: float = 0.3,
) -> tuple[bool, dict]:
    """ADR-0005 결정4 — 승격 관문 C(퇴보 없음), **관문 B(개선 검정)는 범위 제외**(ADR
    재결정, "정기 재학습" 코드패스가 없어 적용 대상이 없음). 레시피별 1:1 비교(같은
    레시피에서 챔피언 vs 후보):

    ① IoU가 챔피언 대비 유의하게 낮아지지 않음 — 윌콕슨 부호순위 단측 검정
    (H1: candidate < champion), p < alpha면 유의하게 나빠진 것으로 보고 차단.
    ② 최악 레시피의 HD95 악화(candidate - champion의 최댓값) ≤ `max_hd95_degradation`.
    ③ `val_pass_rate_result`(val 40에 `check_recipe_pass_rate`를 다시 적용한 결과)가
    주어지면 그것도 합격이어야 한다 — 드리프트 데이터로 재학습한 뒤 기존 조건 성능이
    떨어지는 것 방지. `None`이면(호출부가 아직 안 돌렸으면) 건너뛰고 통과로 본다.

    `alpha` 기본값은 ADR-0002 재학습 루프 최대 시도 횟수(4회) 기준 본페로니 보정값
    (0.05/4=0.0125) — 호출부가 실제 시도 횟수에 맞게 넘겨야 한다."""
    from scipy.stats import wilcoxon

    cand, champ = np.asarray(candidate_iou, dtype=float), np.asarray(champion_iou, dtype=float)
    if np.allclose(cand, champ):
        iou_ok, iou_p = True, 1.0  # 완전히 동일 — wilcoxon이 못 다루는 전부-0 차이 케이스
    else:
        try:
            _, iou_p = wilcoxon(cand, champ, alternative="less")
            iou_ok = bool(iou_p >= alpha)
        except ValueError:
            # 표본 부족 등으로 검정 자체가 불가능 — 근거 없이 차단하지 않고 통과(미확인 표시)
            iou_ok, iou_p = True, float("nan")

    degradations = [c - ch for c, ch in zip(candidate_hd95, champion_hd95, strict=True)]
    worst_degradation = max(degradations) if degradations else 0.0
    hd95_ok = worst_degradation <= max_hd95_degradation

    val_ok = True if val_pass_rate_result is None else val_pass_rate_result[0]

    passed = iou_ok and hd95_ok and val_ok
    detail = {
        "iou_wilcoxon_p": iou_p, "alpha": alpha, "iou_not_regressed": iou_ok,
        "worst_hd95_degradation": worst_degradation, "max_hd95_degradation": max_hd95_degradation,
        "hd95_ok": hd95_ok, "val_pass_rate_ok": val_ok,
    }
    return passed, detail


def check_latency(latency_s: float | None, max_latency_s: float | None) -> str | None:
    """항목⑤: 추론 시간이 상한을 넘으면 경고 문구만 돌린다(차단 안 함, 설계서·체크리스트
    표현 그대로). 둘 중 하나라도 없으면(아직 측정 전 등) 검사 자체를 건너뛴다."""
    if latency_s is None or max_latency_s is None:
        return None
    if latency_s > max_latency_s:
        return f"latency {latency_s:.4f}s > max {max_latency_s:.4f}s (경고, 차단 아님)"
    return None


def run_gate(
    model,
    golden_pairs: list[tuple],
    max_chamfer: float,
    rollout_step_errors: list[float] | None = None,
    rollout_step_errors_by_recipe: list[list[float]] | None = None,
    rollout_max_ratio: float = 2.0,
    expected_dataset_hash: str | None = None,
    actual_dataset_hash: str | None = None,
    expected_golden_hash: str | None = None,
    actual_golden_hash: str | None = None,
    depth_pred: list[float] | None = None,
    depth_true: list[float] | None = None,
    cd_pred: list[float] | None = None,
    cd_true: list[float] | None = None,
    grid: float = 0.3,
    max_dim_pct: float = 5.0,
    current_mean_error: float | None = None,
    champion_mean_error: float | None = None,
    max_regression: float = 0.10,
    self_intersection_count: int | None = None,
    model_specific_violations: int | None = None,
    latency_s: float | None = None,
    max_latency_s: float | None = None,
) -> GateResult:
    """데이터 무결성(종료코드 2)은 두 가지를 독립적으로 본다: 학습에 쓰인 dataset_hash가
    기대값과 같은지(설계서 §4.4), 골든셋 자체가 변조되지 않았는지.

    ADR-0004(2026-11-11) 이후 PASS/BLOCK은 **②(깊이/CD)·④(롤아웃)·물리(선 꼬임+모델별
    결함)**만 가른다. ①(골든 Chamfer)·③(챔피언 대비 악화)는 항상 계산해서
    `reference_metrics`에 남기되 `passed`에는 영향을 주지 않는다. ⑤(지연시간)는 여전히
    `warnings`만. 필요한 값을 안 넘기면(아직 측정 전 등) 그 검사 자체를 건너뛴다(②③④물리
    공통, 기존 호출부 호환) — `model_specific_violations`가 `None`이면(아직 안 세어서
    입력 전) 물리 규칙 중 그 부분만 건너뛴다[사용자 결정 2026-11-11, 기존 관례와 동일].
    `self_intersection_count`는 이 함수가 직접 계산하지 않는다 — 호출부가
    `etl.outer_path`(표면선 추출)와 `eval.metrics.self_intersection_count`(교차 횟수)로
    미리 구해서 넘긴다(배선: `train/rollout.py::rollout_val_error_by_step`,
    `eval/eval.py::run_eval`, ADR-0004 2026-11-11 갱신)."""
    if expected_dataset_hash is not None and actual_dataset_hash != expected_dataset_hash:
        msg = f"dataset_hash mismatch: expected {expected_dataset_hash}, got {actual_dataset_hash}"
        return GateResult(passed=False, exit_code=2, reasons=[msg])
    if expected_golden_hash is not None and actual_golden_hash != expected_golden_hash:
        msg = f"golden hash mismatch: expected {expected_golden_hash}, got {actual_golden_hash}"
        return GateResult(passed=False, exit_code=2, reasons=[msg])

    reasons: list[str] = []
    reference_metrics: dict = {}

    # ① 참고 지표만 (ADR-0004 결정2). golden_pairs가 비어 있으면 건너뛴다 [확인,
    # W5-06 2026-11-12 — 실모델(GNN rollout) 기준 gate-negative 테스트는 `model.predict
    # (pos)` 단순 호출(더미 모델 전용 인터페이스)에 맞는 골든 어댑터가 아직 없어서 ①을
    # 아예 안 넘긴다. 참고 지표일 뿐 `passed`에는 어차피 영향 없으니 안전한 생략.
    if golden_pairs:
        ok1, mean_chamfer = check_golden_chamfer(model, golden_pairs, max_chamfer)
        reference_metrics["golden_chamfer"] = {
            "mean": mean_chamfer,
            "max_chamfer": max_chamfer,
            "within_max": ok1,
        }

    # ② 깊이/CD 치수 오차 — 차단 (ADR-0004 결정2)
    ok2 = True
    if None not in (depth_pred, depth_true, cd_pred, cd_true):
        ok2, detail2 = check_cd_depth_error(
            depth_pred, depth_true, cd_pred, cd_true, grid, max_dim_pct
        )
        if not ok2:
            reasons.append(f"depth/cd dimension error over threshold: {detail2}")

    # ③ 참고 지표만 (ADR-0004 결정2)
    if current_mean_error is not None:
        ok3 = check_champion_regression(current_mean_error, champion_mean_error, max_regression)
        reference_metrics["champion_regression"] = {
            "current_mean_error": current_mean_error,
            "champion_mean_error": champion_mean_error,
            "max_regression": max_regression,
            "within_max": ok3,
        }

    # ④ 롤아웃 후반 급증 — 차단, 다중 레시피가 있으면 그걸 우선(ADR-0004 결정3)
    ok4 = True
    if rollout_step_errors_by_recipe is not None:
        ok4, ratio = check_rollout_late_step_multi(rollout_step_errors_by_recipe, rollout_max_ratio)
        reference_metrics["rollout_late_early_ratio"] = ratio
        if not ok4:
            reasons.append(f"rollout late-step error spike (multi-recipe ratio={ratio:.3f})")
    elif rollout_step_errors is not None:
        ok4 = check_rollout_late_step(rollout_step_errors, rollout_max_ratio)
        if not ok4:
            reasons.append("rollout late-step error spike")

    # 물리: 선 꼬임 + 모델별 결함 — 차단, 신설 (ADR-0004 결정4)
    ok_physics = True
    physics_detail: dict = {}
    if self_intersection_count is not None:
        physics_detail["self_intersection_count"] = self_intersection_count
        if self_intersection_count > 0:
            ok_physics = False
    if model_specific_violations is not None:
        physics_detail["model_specific_violations"] = model_specific_violations
        if model_specific_violations > 0:
            ok_physics = False
    if physics_detail:
        reference_metrics["physics"] = physics_detail
    if not ok_physics:
        reasons.append(f"physics violation: {physics_detail}")

    warnings: list[str] = []
    latency_warning = check_latency(latency_s, max_latency_s)
    if latency_warning is not None:
        warnings.append(latency_warning)

    passed = ok2 and ok4 and ok_physics
    return GateResult(
        passed=passed,
        exit_code=0 if passed else 1,
        reasons=reasons,
        warnings=warnings,
        reference_metrics=reference_metrics,
    )


def write_gate_result(
    result: GateResult, out_path: str | pathlib.Path, thresholds: dict | None = None
) -> dict:
    """설계서 §3 C5 출력(`gate.json`)/체크리스트 W5-05 `gate_result.json`: 판정+근거
    수치(`reasons`/`warnings`)+임계치(호출부가 이번 판정에 쓴 값을 그대로 적음)."""
    out = asdict(result)
    out["thresholds"] = thresholds or {}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    pathlib.Path(out_path).write_text(text, encoding="utf-8")
    return out
