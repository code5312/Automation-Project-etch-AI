"""W6-12: 모델 A/B 비교표 — 같은 test 분할·같은 지표(Chamfer/깊이오차)·같은 조건
(champion fix_mask+guard+loop_removal, FNO는 그대로)으로 비교한다.

두 모델 다 같은 "정답"(outer_path로 추출한 표면선, `rollout_val_error_by_step`/
`run_baselines.py`와 같은 관례)을 기준으로 Chamfer/깊이오차를 낸다 — 모델 B가 자체
SDF 0등고선으로 정답을 다시 뽑으면 추출 방식이 달라 공정한 비교가 아니게 된다.
"""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.etl.etl import outer_path  # noqa: E402
from etchpipe.eval.eval import load_model_a  # noqa: E402
from etchpipe.eval.metrics import chamfer_distance, depth_error  # noqa: E402
from etchpipe.models.fno import FNOModel  # noqa: E402
from etchpipe.train.loop import load_graph  # noqa: E402
from etchpipe.train.rollout import lines_from_edge_index, rollout  # noqa: E402

DATASET_HASH = "4b8c0ce1bc98"
GRAPHS_ROOT = REPO_ROOT / "data" / "processed" / DATASET_HASH / "graphs"
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"
MGN_TUNED_CKPT = REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"
FNO_RUN_DIR = REPO_ROOT / "runs" / "fno_tuned"


def _test_recipe_ids(n: int | None) -> list[str]:
    splits = json.loads(SPLITS_PATH.read_text())
    ids = sorted(splits["test"])
    return ids if n is None else ids[:n]


def _true_curve(rid: str, actual_k: int) -> tuple:
    true_graph = load_graph(GRAPHS_ROOT / rid / f"g{actual_k:03d}.npz")
    true_lines = lines_from_edge_index(true_graph["edge_index"])
    curve = outer_path(true_graph["pos"].numpy(), true_lines)
    return curve


def _model_a_curve(model, norm, rid: str, actual_k: int, step_min: float):
    g0 = load_graph(GRAPHS_ROOT / rid / "g000.npz")
    pred_lines = lines_from_edge_index(g0["edge_index"])
    positions = rollout(
        model, g0["node"], g0["edge_index"], g0["edge"], g0["pos"], actual_k, step_min, norm,
        fix_mask=True, guard=True,
    )
    return outer_path(positions[-1].numpy(), pred_lines)


def main(n_recipes: int | None = None) -> int:
    model_a, note_a, norm_a = load_model_a(MGN_TUNED_CKPT)
    print(f"model A: {note_a}")
    model_b = FNOModel(FNO_RUN_DIR)

    rows = []
    for rid in _test_recipe_ids(n_recipes):
        meta = json.loads((RAW_DIR / rid / "meta.json").read_text())
        actual_k, step_min, params = meta["actual_k"], meta["step_min"], meta["params"]
        true_curve = _true_curve(rid, actual_k)

        curve_a = _model_a_curve(model_a, norm_a, rid, actual_k, step_min)
        vtp0 = RAW_DIR / rid / "snapshots" / "t000.vtp"
        curve_b = model_b.predict_curve(vtp0, params, n_steps=actual_k)

        row = {
            "run_id": rid,
            "actual_k": actual_k,
            "A_chamfer": chamfer_distance(curve_a, true_curve),
            "A_depth_error": depth_error(curve_a, true_curve),
            "B_chamfer": chamfer_distance(curve_b, true_curve),
            "B_depth_error": depth_error(curve_b, true_curve),
        }
        rows.append(row)
        print(f"{rid}: A chamfer={row['A_chamfer']:.3f} depth={row['A_depth_error']:.3f} | "
              f"B chamfer={row['B_chamfer']:.3f} depth={row['B_depth_error']:.3f}")

    def _mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "n": len(rows),
        "A_chamfer_mean": _mean("A_chamfer"),
        "A_depth_error_mean": _mean("A_depth_error"),
        "B_chamfer_mean": _mean("B_chamfer"),
        "B_depth_error_mean": _mean("B_depth_error"),
    }
    print(json.dumps(summary, indent=1))

    out = REPO_ROOT / "reports" / "eval" / "W6-12_model_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(n))
