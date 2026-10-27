import json
import pathlib
import sys
import time

import numpy as np
import pyvista as pv

sys.path.insert(0, "src/etchpipe/sim")
from run_sim import run_sim

sys.path.insert(0, "src/etchpipe/etl")
from etl import load_snapshot, to_graph

import torch
from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch_geometric.data import Data

torch.manual_seed(0)
torch.set_num_threads(4)

THETA_KEYS = [
    "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
    "ion_exponent", "mask_width", "mask_height",
]

# S1에서 가장 빨랐던(=시간 상자에 안전한) 레시피 4개를 골라 K를 늘려 재실행
RECIPE_PARAMS = {
    "s3_r0": {"ion_flux": 3.308, "etchant_flux": 1620.6, "oxygen_flux": 1000.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 7.9, "mask_height": 7.8},
    "s3_r1": {"ion_flux": 4.178, "etchant_flux": 4291.0, "oxygen_flux": 1000.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 13.8, "mask_height": 6.9},
    "s3_r2": {"ion_flux": 6.054, "etchant_flux": 1551.0, "oxygen_flux": 1000.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 11.0, "mask_height": 13.6},
    "s3_r3": {"ion_flux": 4.337, "etchant_flux": 2723.0, "oxygen_flux": 1000.0, "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 8.2, "mask_height": 8.2},
}
K = 6

print("=== 1) 시뮬레이션 (K=6) ===", flush=True)
t0 = time.time()
for rid, params in RECIPE_PARAMS.items():
    out = pathlib.Path(f"data/raw/{rid}")
    if (out / "meta.json").exists():
        print("skip", rid)
        continue
    meta = run_sim(params, out, K=K, rays=300, seed=42)
    print(rid, meta["wall_sec"], "s", flush=True)
print("sim total", round(time.time() - t0, 1), "s")

print("=== 2) ETL (그래프 변환) ===", flush=True)
data_per_recipe = {}
for rid, params in RECIPE_PARAMS.items():
    theta = np.array([params[k] for k in THETA_KEYS], np.float32)
    snaps = sorted(pathlib.Path(f"data/raw/{rid}/snapshots").glob("t*.vtp"))
    steps = []
    for k, s in enumerate(snaps):
        P, L, mid, nv = load_snapshot(s)
        nf, ei, ef = to_graph(P, L, mid, nv, theta, k * 1.0)
        steps.append({"pos": P, "node": nf, "edge_index": ei, "edge": ef, "mid": mid, "nv": nv, "lines": L})
    data_per_recipe[rid] = steps
    print(rid, "steps:", [s["pos"].shape[0] for s in steps])

print("=== 3) 소규모 과적합 학습 (전이쌍: step k -> k+1, 전체 사용) ===", flush=True)
m = MeshGraphNet(input_dim_nodes=14, input_dim_edges=3, output_dim=2, processor_size=6, hidden_dim_processor=64, hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64, hidden_dim_node_decoder=64)
opt = torch.optim.Adam(m.parameters(), 1e-3)


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()


pairs = []
for rid, steps in data_per_recipe.items():
    for k in range(len(steps) - 1):
        pairs.append((steps[k], steps[k + 1]))

t0 = time.time()
losses = []
for it in range(300):
    total_loss = 0.0
    for cur, nxt in pairs:
        nf = torch.tensor(cur["node"])
        ef = torch.tensor(cur["edge"])
        ei = torch.tensor(cur["edge_index"])
        pos = torch.tensor(cur["pos"])
        pos_next = torch.tensor(nxt["pos"])
        g = Data(edge_index=ei, num_nodes=nf.shape[0])
        out = m(nf, ef, g)
        loss = chamfer(pos + out, pos_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    losses.append(total_loss / len(pairs))
    if it % 50 == 0 or it == 299:
        print("iter", it, "avg_loss", round(losses[-1], 4), flush=True)
train_sec = time.time() - t0
print("train_sec", round(train_sec, 1), "first_loss", round(losses[0], 4), "last_loss", round(losses[-1], 4))

print("=== 4) 다단계 롤아웃 (연결·법선 고정, 위치만 갱신) ===", flush=True)
rollout_report = {}
with torch.no_grad():
    for rid, steps in data_per_recipe.items():
        cur = steps[0]
        nf0 = torch.tensor(cur["node"])
        ef0 = torch.tensor(cur["edge"])
        ei0 = torch.tensor(cur["edge_index"])
        mat1h0 = nf0[:, 4:6]  # 재료 원핫(고정 — 스파이크 단순화, 연결 유지)
        theta0 = nf0[:, 6:13]  # theta(7, 레시피 고정값이라 원래도 안 변함)
        nv0 = nf0[:, 2:4]  # 법선(고정 — 스파이크 단순화)
        pos = torch.tensor(cur["pos"]).clone()
        g = Data(edge_index=ei0, num_nodes=nf0.shape[0])
        errs = []
        for k in range(1, len(steps)):
            t_feat = torch.full((pos.shape[0], 1), float(k - 1), dtype=torch.float32)
            nf = torch.cat([pos, nv0, mat1h0, theta0, t_feat], dim=1)
            out = m(nf, ef0, g)
            pos = pos + out
            gt = torch.tensor(steps[k]["pos"])
            err = chamfer(pos, gt).item()
            errs.append(round(err, 4))
        rollout_report[rid] = errs
        print(rid, "chamfer per step:", errs, flush=True)

json.dump(
    {"train_first_loss": losses[0], "train_last_loss": losses[-1], "train_sec": train_sec, "rollout": rollout_report},
    open("reports/evidence/W1-S3_result.json", "w"), indent=1,
)
