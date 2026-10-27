"""C2: raw(.vtp) -> processed(그래프 npz + 거리장 npz + 지표). Curator 래핑 전의 순수 파이썬 버전."""
import json, pathlib, sys
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree

def load_snapshot(path):
    m = pv.read(str(path))
    P = np.asarray(m.points)[:, :2].astype(np.float32)
    lines = m.lines.reshape(-1, 3)[:, 1:]                      # (E,2) 점 인덱스 쌍
    mat = np.asarray(m.cell_data["MaterialIds"]).astype(int)    # 선분별 재료
    nrm = np.asarray(m.cell_data["Normals"])[:, :2]
    # 점 단위 특징으로 변환: 점에 닿는 선분들의 평균
    n = len(P); mid = np.zeros(n); cnt = np.zeros(n); nv = np.zeros((n, 2))
    for e, (a, b) in enumerate(lines):
        for v in (a, b):
            mid[v] += mat[e]; cnt[v] += 1; nv[v] += nrm[e]
    mid = (mid/np.maximum(cnt, 1)); nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
    return P, lines, mid, nv.astype(np.float32)

def to_graph(P, lines, mid, nv, theta, t):
    n = len(P)
    ei = np.concatenate([lines, lines[:, ::-1]]).T              # 양방향 엣지 (2,2E)
    rel = P[ei[1]] - P[ei[0]]
    edge_feat = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], 1)  # 3
    mat1h = np.stack([(mid < 5), (mid >= 5)], 1).astype(np.float32)  # 2: 재료 2종 [가정] 0=Si, 10=마스크
    node_feat = np.concatenate([P, nv, mat1h, np.tile(theta, (n, 1)), np.full((n, 1), t, np.float32)], 1)  # 2+2+2+7+1=14
    return node_feat.astype(np.float32), ei.astype(np.int64), edge_feat.astype(np.float32)

def dist_field(P, nx=256, ny=128, xr=(-15, 15), yr=(-30, 12)):
    """부호 없는 거리장(UDF). 부호는 다음 단계 과제 [제안]."""
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    d, _ = cKDTree(P).query(np.stack([gx.ravel(), gy.ravel()], 1))
    return d.reshape(ny, nx).astype(np.float32)

def metrics(P):
    # Si 윗면이 y=0 (MakeTrench 기본). 식각 깊이 = -ymin
    return {"depth": float(-P[:, 1].min()), "ymin": float(P[:, 1].min())}

def etl_run(raw_dir, out_dir, theta_keys):
    raw = pathlib.Path(raw_dir); meta = json.load(open(raw/"meta.json"))
    theta = np.array([meta["params"][k] for k in theta_keys], np.float32)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    snaps = sorted((raw/"snapshots").glob("t*.vtp")); rows = []
    for k, s in enumerate(snaps):
        P, L, mid, nv = load_snapshot(s)
        nf, ei, ef = to_graph(P, L, mid, nv, theta, k*meta["step_min"])
        np.savez_compressed(out/f"g{k:03d}.npz", node=nf, edge_index=ei, edge=ef, pos=P)
        np.save(out/f"udf{k:03d}.npy", dist_field(P))
        rows.append({"k": k, "n_nodes": int(len(P)), **metrics(P)})
    json.dump(rows, open(out/"metrics.json", "w"), indent=1)
    return rows

if __name__ == "__main__":
    keys = ["ion_flux","etchant_flux","oxygen_flux","ion_mean_energy","ion_exponent","mask_width","mask_height"]
    rid = sys.argv[1]
    rows = etl_run(f"data/raw/{rid}", f"data/processed/{rid}", keys)
    print(rows[0], "...", rows[-1])
