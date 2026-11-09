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

from etchpipe.etl.etl import THETA_KEYS, load_snapshot
from etchpipe.eval.metrics import chamfer_distance, depth_error
from etchpipe.models.baseline import DepthMlpExtrapolationModel
from etchpipe.models.dummy import ZeroDisplacementModel


def _load_run(raw_dir: pathlib.Path, run_id: str) -> tuple[dict, np.ndarray, np.ndarray]:
    run = raw_dir / run_id
    meta = json.loads((run / "meta.json").read_text())
    pos0, *_ = load_snapshot(run / "snapshots" / "t000.vtp")
    k = meta["actual_k"]
    pos_final, *_ = load_snapshot(run / "snapshots" / f"t{k:03d}.vtp")
    return meta, pos0, pos_final


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
        meta, pos0, pos_final = _load_run(raw_dir, rid)
        pred_a = model_a.predict(pos0)
        pred_b = model_b.predict(pos0, meta["params"])
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
