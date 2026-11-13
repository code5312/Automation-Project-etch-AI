"""W6-06: 재학습 전/후 오차 표 — 드리프트 평가셋(golden_drift+val, 20개)에서 재학습 전
챔피언(`runs/mgn_tuned/model.pt`)과 재학습 후 승격 모델을 같은 조건(게이트가 실제로 쓴
`gate_fn`과 동일한 rollout 설정: check_physics=True, fix_mask=True, guard=True,
loop_removal=True)으로 비교한다.

`gate.py::run_gate`의 `reference_metrics`는 ②(깊이/CD)가 통과하면 그 수치를 남기지
않는다(차단 전용 설계, ADR-0004) — 그래서 이번 실제 실행(`retrain_20261112T212130.json`)
에는 깊이/CD 오차의 실제 숫자가 없다. 이 스크립트는 같은 `rollout_val_error_by_step`을
독립적으로 다시 호출해 그 숫자를 복원한다(게이트 판정 자체를 다시 하는 게 아니라 보고용).

사용: PYTHONPATH=src python scripts/compare_retrain_before_after.py
출력: reports/evidence/W6-06_retrain_before_after.json
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from etchpipe.eval.eval import load_model_a
from etchpipe.gate.gate import check_dimension_error, check_rollout_late_step_multi
from etchpipe.retrain.execute import DRIFT_RAW_DIR, EVAL_STEPS, DriftState
from etchpipe.train.rollout import rollout_val_error_by_step

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEFORE_CKPT = ROOT / "runs" / "mgn_tuned" / "model.pt"
AFTER_CKPT = ROOT / "runs" / "retrain" / "mgn_retrain_20261112T212130" / "model.pt"


def _eval_one(ckpt_path: pathlib.Path, state: DriftState) -> dict:
    model, note, norm = load_model_a(ckpt_path)
    chamfer_final, depth_pred, depth_true, chamfer_by_step, self_x_total = [], [], [], [], 0
    for rid in state.eval_ids():
        meta = json.loads((DRIFT_RAW_DIR / rid / "meta.json").read_text())
        actual_k = meta["actual_k"]
        steps = sorted({s for s in EVAL_STEPS if s <= actual_k} | {actual_k})
        result = rollout_val_error_by_step(
            model, state.graphs_root, rid, steps, meta["step_min"], norm=norm,
            check_physics=True, fix_mask=True, guard=True, loop_removal=True,
        )
        chamfer_by_step.append([result[s]["chamfer"] for s in steps])
        final = result[actual_k]
        chamfer_final.append(final["chamfer"])
        depth_pred.append(final["depth_pred"])
        depth_true.append(final["depth_true"])
        self_x_total += final["self_intersections"]

    depth_ok, depth_detail = check_dimension_error(depth_pred, depth_true, grid=0.3, max_pct=5.0)
    ratio_ok, ratio = check_rollout_late_step_multi(chamfer_by_step, max_ratio=2.0)
    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_note": note,
        "n_eval": len(state.eval_ids()),
        "chamfer_mean": float(np.mean(chamfer_final)),
        "chamfer_median": float(np.median(chamfer_final)),
        "depth_error": depth_detail,
        "depth_dimension_ok": depth_ok,
        "rollout_late_early_ratio": ratio,
        "rollout_ratio_ok": ratio_ok,
        "self_intersection_count": self_x_total,
    }


def main() -> int:
    print("드리프트 평가셋(golden_drift+val) 로드·ETL 중...")
    state = DriftState()
    print(f"  eval_ids={len(state.eval_ids())}개, dataset_hash={state.dataset_hash}")

    print(f"재학습 전({BEFORE_CKPT}) 평가 중...")
    before = _eval_one(BEFORE_CKPT, state)
    print("  ->", json.dumps(before, indent=2, ensure_ascii=False))

    print(f"재학습 후({AFTER_CKPT}) 평가 중...")
    after = _eval_one(AFTER_CKPT, state)
    print("  ->", json.dumps(after, indent=2, ensure_ascii=False))

    summary = {"dataset_hash": state.dataset_hash, "before": before, "after": after}
    out_path = ROOT / "reports" / "evidence" / "W6-06_retrain_before_after.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
