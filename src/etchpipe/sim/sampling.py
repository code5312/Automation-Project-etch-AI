"""C1-a: 레시피 파라미터 Θ 샘플링 (Sobol). 결과: data/recipes.json"""
import json, sys
import numpy as np
from scipy.stats import qmc

# (이름, 최소, 최대, log 스케일 여부) -- 범위는 [제안]. 1주차 스파이크에서 확정.
SPACE = [
    ("ion_flux",        3.0,   30.0,  True),   # 기본값 10 의 x0.3~x3
    ("etchant_flux",    1350., 13500., True),  # 기본 4500
    ("oxygen_flux",     240.,  2400., True),   # passivationFlux, 기본 800
    ("ion_mean_energy", 50.,   200.,  False),
    ("ion_exponent",    100.,  1000., True),
    ("mask_width",      6.,    14.,   False),  # trenchWidth (um)
    ("mask_height",     6.,    14.,   False),
]

def make_recipes(n_total, seed=0, base_two=True):
    d = len(SPACE)
    u = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n_total))))[:n_total]
    out = []
    for i, row in enumerate(u):
        p = {}
        for (name, lo, hi, log), x in zip(SPACE, row):
            p[name] = float(np.exp(np.log(lo) + x*(np.log(hi)-np.log(lo))) if log else lo + x*(hi-lo))
        out.append({"run_id": f"r{i:04d}", "params": p})
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rec = make_recipes(n)
    json.dump(rec, open("data/recipes.json", "w"), indent=1)
    print(f"{len(rec)} recipes -> data/recipes.json")
