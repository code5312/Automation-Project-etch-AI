"""W3-06: 골든셋 확정.

train split(300건)에서 `select_golden_recipes`(중심5·경계3·극단2)로 10개를 고르고,
`data/raw/<run_id>/`를 `golden/raw/<run_id>/`로 복사한다. `golden/golden.json`에
선정 결과(run_id·params·role)와 `golden_hash`(각 run meta.json의 params_hash를
모은 sha1, compute_dataset_hash와 동일한 방식)를 기록한다.
재생성 금지(설계서 §4.1) — 이 스크립트는 한 번만 실행하고, 이후 golden/raw 변경은
의도적 변경(ADR)이 아니면 테스트(test_golden.py)가 golden_hash 불일치로 잡아낸다.

사용: python scripts/make_golden.py [recipes.json] [splits.json] [raw_dir] [out_dir]
"""

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from etchpipe.etl.etl import compute_dataset_hash  # noqa: E402
from etchpipe.sim.sampling import select_golden_recipes  # noqa: E402


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    recipes_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/recipes.json"
    splits_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data/splits.json"
    raw_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else root / "data/raw"
    out_dir = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else root / "golden"

    recipes = {r["run_id"]: r for r in json.loads(recipes_path.read_text())}
    splits = json.loads(splits_path.read_text())
    train_recipes = [recipes[rid] for rid in splits["train"]]

    n_center, n_edge, n_extreme = 5, 3, 2
    picked = select_golden_recipes(train_recipes, n_center, n_edge, n_extreme)
    roles = ["center"] * n_center + ["edge"] * n_edge + ["extreme"] * n_extreme

    golden_raw = out_dir / "raw"
    golden_raw.mkdir(parents=True, exist_ok=True)
    selected = []
    for role, recipe in zip(roles, picked):
        rid = recipe["run_id"]
        src = raw_dir / rid
        dst = golden_raw / rid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        selected.append({"run_id": rid, "role": role, "params": recipe["params"]})

    run_ids = sorted(s["run_id"] for s in selected)
    golden_hash = compute_dataset_hash(run_ids, golden_raw)
    manifest = {
        "rule": "center5_edge3_extreme2",
        "source_split": "train",
        "run_ids": run_ids,
        "selected": selected,
        "golden_hash": golden_hash,
    }
    (out_dir / "golden.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"golden/raw: {len(run_ids)}건 복사, golden_hash={golden_hash}")
    for s in selected:
        print(" ", s["role"], s["run_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
