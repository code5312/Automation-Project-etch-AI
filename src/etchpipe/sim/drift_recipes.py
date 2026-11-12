"""W6-03: 드리프트 데이터 60개(train 40/val 10/golden_drift 10) 레시피 분할 —
`run_sim`(viennaps 필요) 없이 Θ 샘플링만 한다(`sampling.py`와 같은 의존성 수준). 이렇게
분리해야 `scripts/process_drift_data.py`(train 컨테이너)가 이 함수를 쓰면서도
`run_drift_data_gen.py`를 거쳐 viennaps를 끌어오지 않는다."""

from etchpipe.sim.sampling import make_recipes

N_TRAIN, N_VAL, N_GOLDEN_DRIFT = 40, 10, 10
SEED = 9100
PREFIX = "drift"


def drift_splits() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """train 40 / val 10 / golden_drift 10 — 한 번의 Sobol 추출을 순서대로 자른다
    (sampling.make_splits과 같은 패턴)."""
    recipes = make_recipes(N_TRAIN + N_VAL + N_GOLDEN_DRIFT, seed=SEED, prefix=PREFIX)
    splits = {
        "train": [r["run_id"] for r in recipes[:N_TRAIN]],
        "val": [r["run_id"] for r in recipes[N_TRAIN : N_TRAIN + N_VAL]],
        "golden_drift": [r["run_id"] for r in recipes[N_TRAIN + N_VAL :]],
    }
    params_by_id = {r["run_id"]: r["params"] for r in recipes}
    return splits, params_by_id
