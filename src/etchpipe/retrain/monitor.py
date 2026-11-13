"""W6-02: 드리프트 모니터(설계서 §3 C8, ADR-0002) — FR-09.

ADR-0005 결정5(2026-11-14): 트리거 판정을 "champion의 골든셋 Chamfer × threshold_mult
초과" 방식에서 **결정3(레시피별 깊이·CD·롤아웃 3조건 통과율)** 방식으로 교체했다 —
드리프트 레시피에 그 3조건을 적용해 챔피언의 통과율이 90%(9/10) 미만이면 트리거.
`mean_chamfer_over_recipes`는 참고 지표 계산용으로 그대로 남겨둔다(다른 호출부 없음,
Chamfer 기반 트리거만 폐기).

ViennaPS 시뮬레이션·그래프 변환(etl_run) 자체는 이 모듈 밖(호출부, 예: W6-02 실행
스크립트)에서 끝내고, 이 모듈은 이미 변환된 그래프(`graphs_root/<run_id>/g*.npz`)만
받는다 — 그래야 ViennaPS 없는 환경(train 컨테이너, 단위 테스트)에서도 로직을 그대로
테스트할 수 있다(etl.py/gate.py와 같은 분리 패턴)."""

import json
import pathlib

from etchpipe.gate.gate import check_recipe_pass_rate
from etchpipe.train.rollout import rollout_val_error_by_step

EVAL_STEPS = (5, 10)  # retrain/execute.py와 동일 — 롤아웃 조기/후기 비교용 중간 스텝


def _final_chamfer(
    model,
    norm,
    graphs_root: pathlib.Path,
    raw_dir: pathlib.Path,
    run_id: str,
    fix_mask: bool,
    guard: bool,
    loop_removal: bool,
) -> float:
    meta = json.loads((pathlib.Path(raw_dir) / run_id / "meta.json").read_text())
    actual_k = meta["actual_k"]
    result = rollout_val_error_by_step(
        model, graphs_root, run_id, [actual_k], meta["step_min"], norm=norm,
        fix_mask=fix_mask, guard=guard, loop_removal=loop_removal,
    )
    return result[actual_k]["chamfer"]


def mean_chamfer_over_recipes(
    model,
    norm,
    graphs_root: pathlib.Path,
    raw_dir: pathlib.Path,
    run_ids: list[str],
    fix_mask: bool = True,
    guard: bool = True,
    loop_removal: bool = True,
) -> dict:
    """`run_ids` 각각의 최종 스텝 Chamfer(champion 예측 vs 실제 시뮬레이션 결과)를 구해
    평균·개별값을 낸다. `champion_golden_chamfer`(①의 기준값)와 `check_drift`(모니터링
    레시피 판정) 둘 다 이 함수 하나로 계산한다 — 같은 지표를 어느 레시피 집합에 쓰느냐만
    다르다."""
    per_recipe = {
        rid: _final_chamfer(model, norm, graphs_root, raw_dir, rid, fix_mask, guard, loop_removal)
        for rid in run_ids
    }
    return {"mean": sum(per_recipe.values()) / len(per_recipe), "per_recipe": per_recipe}


def _recipe_eval(
    model,
    norm,
    graphs_root: pathlib.Path,
    raw_dir: pathlib.Path,
    run_id: str,
    fix_mask: bool,
    guard: bool,
    loop_removal: bool,
) -> dict:
    """레시피 1개의 깊이(마지막 스텝)·CD(아직 안 배선, 깊이로 대체 — `gate/execute.py`
    와 동일한 [미확인, 알려진 한계])·롤아웃 스텝별 Chamfer(조기/후기 비교용)를 구한다."""
    meta = json.loads((pathlib.Path(raw_dir) / run_id / "meta.json").read_text())
    actual_k = meta["actual_k"]
    steps = sorted({s for s in EVAL_STEPS if s <= actual_k} | {actual_k})
    result = rollout_val_error_by_step(
        model, graphs_root, run_id, steps, meta["step_min"], norm=norm,
        fix_mask=fix_mask, guard=guard, loop_removal=loop_removal,
    )
    final = result[actual_k]
    return {
        "depth_pred": final["depth_pred"],
        "depth_true": final["depth_true"],
        "rollout_step_errors": [result[s]["chamfer"] for s in steps],
    }


def check_drift(
    model,
    norm,
    monitor_graphs_root: pathlib.Path,
    monitor_raw_dir: pathlib.Path,
    monitor_run_ids: list[str],
    grid: float = 0.3,
    max_dim_pct: float = 5.0,
    cd_floor: float = 0.6,
    rollout_max_ratio: float = 2.5,
    rollout_min_increase: float = 0.3,
    pass_rate: float = 0.9,
    fix_mask: bool = True,
    guard: bool = True,
    loop_removal: bool = True,
) -> dict:
    """ADR-0005 결정5 — 드리프트 레시피(`monitor_run_ids`, 설계서 §4.1 기준 10개)에
    `check_recipe_pass_rate`(결정3, 깊이·CD·롤아웃 3조건)를 적용해 챔피언의 통과율이
    `pass_rate`(기본 90%, 10개 중 9개) 미만이면 `triggered=True`(ADR-0002 자동 반복
    루프 트리거 신호 — 이 함수는 판정만 하고 루프 실행은 W6-04가 함).

    레시피 10개뿐이라 통계적 검정력이 낮다(문서 §10: 9/10 충족 시 95% 신뢰하한 0.61) —
    **트리거 신호로만 쓰고, 승격 판정 자체는 20개 이상(golden_drift10+val10)으로
    별도 판정한다**(`gate.check_recipe_pass_rate`/`check_no_regression`, retrain
    루프 쪽)."""
    evals = [
        _recipe_eval(
            model, norm, monitor_graphs_root, monitor_raw_dir, rid, fix_mask, guard, loop_removal
        )
        for rid in monitor_run_ids
    ]
    depth_pred = [e["depth_pred"] for e in evals]
    depth_true = [e["depth_true"] for e in evals]
    # [미확인, 알려진 한계 — gate.py/retrain/execute.py와 동일] CD 아직 안 배선, 깊이로 대체.
    cd_pred, cd_true = depth_pred, depth_true
    rollout_errors = [e["rollout_step_errors"] for e in evals]

    ok, detail = check_recipe_pass_rate(
        depth_pred, depth_true, cd_pred, cd_true, rollout_errors,
        grid=grid, max_dim_pct=max_dim_pct, cd_floor=cd_floor,
        rollout_max_ratio=rollout_max_ratio, rollout_min_increase=rollout_min_increase,
        pass_rate=pass_rate,
    )
    return {
        "n": detail["n"],
        "n_passed": detail["n_passed"],
        "pass_rate": detail["pass_rate"],
        "required_pass_rate": pass_rate,
        "per_recipe": detail["per_recipe"],
        "triggered": not ok,
    }
