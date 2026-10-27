import json
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, "src/etchpipe/sim")
from run_sim import run_sim

params = json.load(open("data/raw/r0000/meta.json"))["params"]

# 같은 레시피·같은 seed로 한 번 더 실행 (K=2, rays=300 — S1과 동일 조건)
meta2 = run_sim(params, "data/raw/r0000_repeat", K=2, rays=300, seed=42)
print("meta2:", meta2)

for k in (0, 1, 2):
    a = pv.read(f"data/raw/r0000/snapshots/t{k:03d}.vtp")
    b = pv.read(f"data/raw/r0000_repeat/snapshots/t{k:03d}.vtp")
    pa, pb = np.asarray(a.points), np.asarray(b.points)
    same_n = pa.shape[0] == pb.shape[0]
    print(f"t{k:03d}: n_points a={pa.shape[0]} b={pb.shape[0]} same_n={same_n}", end=" ")
    if same_n:
        diff = np.abs(pa - pb).max()
        print(f"max_abs_diff={diff:.6g}")
    else:
        print("(점 개수 다름 - 좌표 직접 비교 불가)")

depth_a = -np.asarray(pv.read("data/raw/r0000/snapshots/t002.vtp").points)[:, 1].min()
depth_b = -np.asarray(pv.read("data/raw/r0000_repeat/snapshots/t002.vtp").points)[:, 1].min()
print("depth a:", depth_a, "depth b:", depth_b, "diff:", abs(depth_a - depth_b))
