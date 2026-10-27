"""C2: raw(.vtp) -> processed(그래프 npz + 거리장 npz + 지표 + dataset_hash/manifest).

Curator 래핑 전의 순수 파이썬 버전. SDF(부호 있는 거리장)는 3주차(W3-03)에서 완성한다.
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

SnapshotArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_snapshot(path: str | pathlib.Path) -> SnapshotArrays:
    m = pv.read(str(path))
    p = np.asarray(m.points)[:, :2].astype(np.float32)
    lines = m.lines.reshape(-1, 3)[:, 1:]  # (E,2) 점 인덱스 쌍
    mat = np.asarray(m.cell_data["MaterialIds"]).astype(int)  # 선분별 재료
    nrm = np.asarray(m.cell_data["Normals"])[:, :2]
    # 점 단위 특징으로 변환: 점에 닿는 선분들의 평균
    n = len(p)
    mid = np.zeros(n)
    cnt = np.zeros(n)
    nv = np.zeros((n, 2))
    for e, (a, b) in enumerate(lines):
        for v in (a, b):
            mid[v] += mat[e]
            cnt[v] += 1
            nv[v] += nrm[e]
    mid = mid / np.maximum(cnt, 1)
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
    return p, lines, mid, nv.astype(np.float32)


def to_graph(
    p: np.ndarray, lines: np.ndarray, mid: np.ndarray, nv: np.ndarray, theta: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(p)
    ei = np.concatenate([lines, lines[:, ::-1]]).T  # 양방향 엣지 (2,2E)
    rel = p[ei[1]] - p[ei[0]]
    edge_feat = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], 1)  # 3
    # 재료 2종 원핫. W1-S5 실측 확정: MaterialIds 0=mask, 10=Si (configs/etl/materials.yaml)
    mat1h = np.stack([(mid < 5), (mid >= 5)], 1).astype(np.float32)  # [0]=mask, [1]=Si
    node_feat = np.concatenate(
        [p, nv, mat1h, np.tile(theta, (n, 1)), np.full((n, 1), t, np.float32)], 1
    )  # 2+2+2+7+1=14
    return node_feat.astype(np.float32), ei.astype(np.int64), edge_feat.astype(np.float32)


def dist_field(
    p: np.ndarray,
    nx: int = 256,
    ny: int = 128,
    xr: tuple[float, float] = (-15, 15),
    yr: tuple[float, float] = (-30, 12),
) -> np.ndarray:
    """부호 없는 거리장(UDF). 부호는 3주차 과제(W3-03) [제안]."""
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    d, _ = cKDTree(p).query(np.stack([gx.ravel(), gy.ravel()], 1))
    return d.reshape(ny, nx).astype(np.float32)


def metrics(p: np.ndarray) -> dict:
    # Si 윗면이 y=0 (MakeTrench 기본). 식각 깊이 = -ymin
    return {"depth": float(-p[:, 1].min()), "ymin": float(p[:, 1].min())}


def etl_run(
    raw_dir: str | pathlib.Path, out_dir: str | pathlib.Path, theta_keys: list[str]
) -> list[dict]:
    """레시피 1건의 raw 스냅샷 전체를 그래프로 변환해 out_dir에 저장."""
    raw = pathlib.Path(raw_dir)
    meta = json.load(open(raw / "meta.json"))
    theta = np.array([meta["params"][k] for k in theta_keys], np.float32)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snaps = sorted((raw / "snapshots").glob("t*.vtp"))
    rows = []
    for k, s in enumerate(snaps):
        p, lines, mid, nv = load_snapshot(s)
        nf, ei, ef = to_graph(p, lines, mid, nv, theta, k * meta["step_min"])
        np.savez_compressed(out / f"g{k:03d}.npz", node=nf, edge_index=ei, edge=ef, pos=p)
        np.save(out / f"udf{k:03d}.npy", dist_field(p))
        rows.append({"k": k, "n_nodes": int(len(p)), **metrics(p)})
    json.dump(rows, open(out / "metrics.json", "w"), indent=1)
    return rows


def compute_dataset_hash(run_ids: list[str], raw_dir: str | pathlib.Path) -> str:
    """각 run의 meta.json params_hash를 모아 데이터셋 전체의 내용 해시를 만든다.
    레시피 하나라도 바뀌면(재생성 포함) 해시가 바뀐다(설계서 §4.4)."""
    parts = []
    for rid in sorted(run_ids):
        meta = json.load(open(pathlib.Path(raw_dir) / rid / "meta.json"))
        parts.append(meta["params_hash"])
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:12]


def process_dataset(
    raw_dir: str | pathlib.Path,
    splits: dict[str, list[str]],
    out_root: str | pathlib.Path,
    theta_keys: list[str],
) -> dict:
    """splits(train/val/test/ood -> run_id 리스트)를 raw_dir에서 읽어
    out_root/<dataset_hash>/graphs/<run_id>/에 쓰고 manifest.json·splits.json을 남긴다."""
    run_ids = sorted({rid for group in splits.values() for rid in group})
    dataset_hash = compute_dataset_hash(run_ids, raw_dir)
    out = pathlib.Path(out_root) / dataset_hash
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        etl_run(pathlib.Path(raw_dir) / rid, out / "graphs" / rid, theta_keys)
    json.dump(splits, open(out / "splits.json", "w"), indent=1)
    manifest = {
        "dataset_hash": dataset_hash,
        "theta_keys": theta_keys,
        "n_recipes": len(run_ids),
        "splits_sizes": {k: len(v) for k, v in splits.items()},
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    return manifest


THETA_KEYS = [
    "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
    "ion_exponent", "mask_width", "mask_height",
]

if __name__ == "__main__":
    rid = sys.argv[1]
    rows = etl_run(f"data/raw/{rid}", f"data/processed/{rid}", THETA_KEYS)
    print(rows[0], "...", rows[-1])
