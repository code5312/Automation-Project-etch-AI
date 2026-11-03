"""W3-01: make_recipes/make_splits 실행 -> data/recipes.json(전체 레시피),
data/splits.json(run_id 분할, ADR-0001 스키마: train/val/test/ood) 생성.

split 크기·seed는 configs/sim/base.yaml의 값을 손으로 맞춰 둔 것(값이 바뀌면 같이 고친다) —
이 스크립트는 PyYAML에 의존하지 않도록 YAML을 직접 읽지 않는다(새 의존성 추가는 사용자 확인 필요,
CLAUDE.md 금지 항목).

골든(10)은 선정 규칙(중심5·경계3·극단2)이 아직 미정이라 포함하지 않음 — W3-06에서 별도 처리.
"""

import json

from etchpipe.sim.sampling import make_splits

SEED = 20261027  # configs/sim/base.yaml: seed
N_TRAIN, N_VAL, N_TEST, N_OOD = 300, 40, 40, 20  # configs/sim/base.yaml: splits


def main() -> None:
    splits = make_splits(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, n_ood=N_OOD, seed=SEED)

    all_recipes = splits["train"] + splits["val"] + splits["test"] + splits["ood"]
    json.dump(all_recipes, open("data/recipes.json", "w"), indent=1)

    splits_ids = {k: [r["run_id"] for r in v] for k, v in splits.items()}
    json.dump(splits_ids, open("data/splits.json", "w"), indent=1)

    print(f"recipes: {len(all_recipes)} -> data/recipes.json")
    print(f"splits sizes: {({k: len(v) for k, v in splits_ids.items()})} -> data/splits.json")


if __name__ == "__main__":
    main()
