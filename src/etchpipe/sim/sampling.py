"""C1-a: 레시피 파라미터 Θ 샘플링(Sobol) + train/val/test/ood 분할.

결과: data/recipes.json(샘플링), splits.json(분할). 분할은 레시피 단위(설계서 §4.1).
OOD는 정의서 §4.1 "학습 범위 바깥에서 따로 생성" — 각 파라미터를 학습 범위 밖(아래 또는 위)으로
벗어나게 뽑는다(설계서 §4.2 확정 전까지 범위는 [제안]).
"""

import json
import sys

import numpy as np
from scipy.stats import qmc

# (이름, 최소, 최대, log 스케일 여부) -- 범위는 [제안]. 1주차 스파이크에서 확정.
SPACE = [
    ("ion_flux", 3.0, 30.0, True),  # 기본값 10 의 x0.3~x3
    ("etchant_flux", 1350.0, 13500.0, True),  # 기본 4500
    ("oxygen_flux", 240.0, 2400.0, True),  # passivationFlux, 기본 800
    ("ion_mean_energy", 50.0, 200.0, False),
    ("ion_exponent", 100.0, 1000.0, True),
    ("mask_width", 6.0, 14.0, False),  # trenchWidth (um)
    ("mask_height", 6.0, 14.0, False),
]


def make_recipes(n_total: int, seed: int = 0, prefix: str = "r") -> list[dict]:
    """Sobol 샘플 n_total개를 SPACE 범위 안에서 뽑는다. 같은 seed -> 같은 결과."""
    d = len(SPACE)
    m = int(np.ceil(np.log2(n_total)))
    u = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)[:n_total]
    out = []
    for i, row in enumerate(u):
        p = {}
        for (name, lo, hi, log), x in zip(SPACE, row):
            if log:
                p[name] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
            else:
                p[name] = float(lo + x * (hi - lo))
        out.append({"run_id": f"{prefix}{i:04d}", "params": p})
    return out


def make_ood_recipes(n_total: int, seed: int = 0, extend: float = 1.5) -> list[dict]:
    """학습 범위(SPACE) 바깥 레시피를 뽑는다. 각 파라미터를 lo 아래 또는 hi 위 구간에서
    균등 샘플링(동전 던지기로 방향 선택, seed로 결정적)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_total):
        p = {}
        for name, lo, hi, log in SPACE:
            below = rng.random() < 0.5
            if log:
                ext_lo, ext_hi = (lo / extend, lo) if below else (hi, hi * extend)
                p[name] = float(np.exp(rng.uniform(np.log(ext_lo), np.log(ext_hi))))
            else:
                span = (hi - lo) * (extend - 1)
                ext_lo, ext_hi = (lo - span, lo) if below else (hi, hi + span)
                p[name] = float(rng.uniform(ext_lo, ext_hi))
        out.append({"run_id": f"ood{i:04d}", "params": p})
    return out


def make_splits(
    n_train: int = 300, n_val: int = 40, n_test: int = 40, n_ood: int = 20, seed: int = 0
) -> dict[str, list[dict]]:
    """레시피 단위 분할. train/val/test는 한 번의 Sobol 추출을 순서대로 자른 것이라
    서로 겹치지 않고(교집합 0), ood는 별도 분포(학습 범위 밖)에서 뽑아 run_id 접두사도 다르다."""
    recipes = make_recipes(n_train + n_val + n_test, seed=seed)
    train = recipes[:n_train]
    val = recipes[n_train : n_train + n_val]
    test = recipes[n_train + n_val :]
    ood = make_ood_recipes(n_ood, seed=seed + 1)
    return {"train": train, "val": val, "test": test, "ood": ood}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rec = make_recipes(n)
    json.dump(rec, open("data/recipes.json", "w"), indent=1)
    print(f"{len(rec)} recipes -> data/recipes.json")
