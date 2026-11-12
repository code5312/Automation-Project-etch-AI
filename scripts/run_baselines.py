"""W3-07: 베이스라인 2종을 val split에서 평가해 지표를 기록한다.

(a) ZeroDisplacementModel: 초기 표면(t=0)을 그대로 최종 예측으로 사용.
(b) DepthMlpExtrapolationModel: train split(300건)의 Θ->최종 깊이로 MLP를 학습한 뒤,
    val split(40건)의 각 레시피에서 초기 표면을 예측 깊이만큼 수직 이동.

두 베이스라인 모두 "최종 표면 1장 예측"(t=0 -> t=actual_k) 기준으로 평가한다 — 체크리스트
D4 판정의 후퇴안(최종 프로파일 1장 예측)과 같은 틀을 미리 맞춰, W4-06에서 모델 A 롤아웃
오차와 바로 비교할 수 있게 한다.

사용: python scripts/run_baselines.py [data_dir]
출력: reports/evidence/W3-07_baseline_metrics.json (레시피별), 표는 evidence .md에 수기 기록.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from etchpipe.etl.etl import THETA_KEYS, load_snapshot, outer_path
from etchpipe.eval.metrics import chamfer_distance, depth_error
from etchpipe.models.baseline import DepthMlpExtrapolationModel
from etchpipe.models.dummy import ZeroDisplacementModel


def _load_run(
    raw_dir: pathlib.Path, run_id: str
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """ADR-0004 후속 수정(2026-11-11): `pos0`는 `outer_path`로 걸러낸 표면선이 아니라
    원본(점 순서 그대로, lines도 같이 반환) — 예측(변위 적용)이 이 원본 점 순서/개수를
    그대로 보존하므로 예측 쪽 `outer_path`는 호출부에서 `pos0`의 `lines`로 한다.
    `pos_final`(정답)은 여기서 바로 `outer_path`로 표면선만 추출해 반환한다."""
    run = raw_dir / run_id
    meta = json.loads((run / "meta.json").read_text())
    pos0, lines0, *_ = load_snapshot(run / "snapshots" / "t000.vtp")
    k = meta["actual_k"]
    pos_final_raw, lines_final, *_ = load_snapshot(run / "snapshots" / f"t{k:03d}.vtp")
    pos_final = outer_path(pos_final_raw, lines_final)
    return meta, pos0, lines0, pos_final


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data"
    raw_dir = data_dir / "raw"
    splits = json.loads((data_dir / "splits.json").read_text())

    train_metas = [_load_run(raw_dir, rid)[0] for rid in splits["train"]]
    thetas = [m["params"] for m in train_metas]
    depths = [m["final_depth_um"] for m in train_metas]

    model_b = DepthMlpExtrapolationModel(THETA_KEYS, hidden=16, seed=0)
    model_b.fit(thetas, depths, epochs=500, lr=0.05)
    model_a = ZeroDisplacementModel()

    per_run = []
    for rid in splits["val"]:
        meta, pos0, lines0, pos_final = _load_run(raw_dir, rid)
        # 예측은 pos0와 같은 점 순서·개수를 보존(변위만 적용)하므로 pos0의 lines로
        # outer_path를 적용해 표면선만 뽑는다(ADR-0004 후속 수정, 2026-11-11).
        pred_a = outer_path(model_a.predict(pos0), lines0)
        pred_b = outer_path(model_b.predict(pos0, meta["params"]), lines0)
        per_run.append(
            {
                "run_id": rid,
                "true_depth": meta["final_depth_um"],
                "a_zero_displacement": {
                    "chamfer": chamfer_distance(pred_a, pos_final),
                    "depth_error": depth_error(pred_a, pos_final),
                },
                "b_depth_mlp_extrapolation": {
                    "chamfer": chamfer_distance(pred_b, pos_final),
                    "depth_error": depth_error(pred_b, pos_final),
                    "pred_depth": model_b.predict_depth(meta["params"]),
                },
            }
        )

    def _agg(key: str, metric: str) -> dict:
        vals = [r[key][metric] for r in per_run]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}

    summary = {
        "n_val": len(per_run),
        "a_zero_displacement": {
            "chamfer": _agg("a_zero_displacement", "chamfer"),
            "depth_error": _agg("a_zero_displacement", "depth_error"),
        },
        "b_depth_mlp_extrapolation": {
            "chamfer": _agg("b_depth_mlp_extrapolation", "chamfer"),
            "depth_error": _agg("b_depth_mlp_extrapolation", "depth_error"),
        },
        "per_run": per_run,
    }

    out_path = root / "reports" / "evidence" / "W3-07_baseline_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
