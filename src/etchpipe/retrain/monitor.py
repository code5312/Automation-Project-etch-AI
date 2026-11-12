"""W6-02: 드리프트 모니터(설계서 §3 C8, ADR-0002) — FR-09.

신규 레시피를 live(드리프트) 설정으로 시뮬레이션한 결과와 champion 예측 사이의
Chamfer를, champion의 골든셋 Chamfer × `threshold_mult`(기본 1.5, `drift_threshold`)와
비교해 자동 반복 루프(ADR-0002, W6-04) 트리거 여부를 판정한다.

ViennaPS 시뮬레이션·그래프 변환(etl_run) 자체는 이 모듈 밖(호출부, 예: W6-02 실행
스크립트)에서 끝내고, 이 모듈은 이미 변환된 그래프(`graphs_root/<run_id>/g*.npz`)만
받는다 — 그래야 ViennaPS 없는 환경(train 컨테이너, 단위 테스트)에서도 로직을 그대로
테스트할 수 있다(etl.py/gate.py와 같은 분리 패턴)."""

import json
import pathlib

from etchpipe.train.rollout import rollout_val_error_by_step


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


def check_drift(
    model,
    norm,
    champion_golden_chamfer: float,
    monitor_graphs_root: pathlib.Path,
    monitor_raw_dir: pathlib.Path,
    monitor_run_ids: list[str],
    threshold_mult: float = 1.5,
    fix_mask: bool = True,
    guard: bool = True,
    loop_removal: bool = True,
) -> dict:
    """체크리스트 W6-02 그대로: `drift_threshold = champion_golden_chamfer *
    threshold_mult` 초과 시 `triggered=True`(ADR-0002 자동 반복 루프 트리거 신호 —
    이 함수는 판정만 하고 루프 실행은 W6-04가 함)."""
    drift_threshold = champion_golden_chamfer * threshold_mult
    result = mean_chamfer_over_recipes(
        model, norm, monitor_graphs_root, monitor_raw_dir, monitor_run_ids,
        fix_mask=fix_mask, guard=guard, loop_removal=loop_removal,
    )
    return {
        "mean_chamfer": result["mean"],
        "per_recipe_chamfer": result["per_recipe"],
        "champion_golden_chamfer": champion_golden_chamfer,
        "threshold_mult": threshold_mult,
        "drift_threshold": drift_threshold,
        "triggered": result["mean"] > drift_threshold,
    }
