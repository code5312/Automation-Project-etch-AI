"""C2: raw(.vtp) -> processed(그래프 npz + 거리장 npz + 지표 + dataset_hash/manifest).

Curator 래핑 전의 순수 파이썬 버전. SDF(부호 있는 거리장)는 W3-03에서 완성.
"""

import hashlib
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

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


# y 범위 자동 결정(W3-03)에 쓰는 값들의 근거:
#  - DEPTH_MAX: configs/sim/base.yaml sim.depth_max(정지조건) [확인]
#  - MASK_HEIGHT_MAX_OOD: sampling.py SPACE의 mask_height hi=14.0에
#    make_ood_recipes의 extend=1.5(= configs/sim/base.yaml ood.extend)를 적용한 최대값
#    14 + (14-6)*(1.5-1) = 18.0 [확인, sampling.make_ood_recipes 공식과 동일 계산]
#  옛 하드코딩 yr=(-30, 12)는 depth_max=50 적용 전 값이라 현재 depth_max를 담지 못해
#  깊은 레시피에서 바닥이 잘리는 버그였다 — 이번에 config 근거로 교체.
DEPTH_MAX = 50.0
MASK_HEIGHT_MAX_OOD = 18.0
Y_MARGIN = 3.0  # 정지조건 오버슈트·그리드 경계 효과 대비 여유 [제안]


def auto_yr(
    depth_max: float = DEPTH_MAX,
    mask_height_max: float = MASK_HEIGHT_MAX_OOD,
    margin: float = Y_MARGIN,
) -> tuple[float, float]:
    """거리장 y 격자 범위를 하드코딩 대신 depth_max·mask_height_max(OOD 포함)에서 자동 결정."""
    return (-(depth_max + margin), mask_height_max + margin)


def dist_field(
    p: np.ndarray,
    nx: int = 256,
    ny: int = 128,
    xr: tuple[float, float] = (-15, 15),
    yr: tuple[float, float] | None = None,
) -> np.ndarray:
    """부호 없는 거리장(UDF, 각 격자점에서 가장 가까운 경계점까지 거리)."""
    if yr is None:
        yr = auto_yr()
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    d, _ = cKDTree(p).query(np.stack([gx.ravel(), gy.ravel()], 1))
    return d.reshape(ny, nx).astype(np.float32)


def _inside_solid(
    x: np.ndarray, y: np.ndarray, p: np.ndarray, lines: np.ndarray, y_below: float
) -> np.ndarray:
    """짝수-홀수 규칙(even-odd rule)으로 격자점마다 solid(마스크+Si) 내부 여부 판정.
    항상 solid인 바닥(y_below)에서 점까지 수직 반직선을 쏴 경계 선분과 교차하는 횟수를 센다
    — 횟수가 짝수면 바닥과 같은 상태(내부), 홀수면 뒤집힌 상태(외부/vacuum).
    언더컷·보잉처럼 한 x에서 경계를 여러 번 지나는 비단조 프로파일도 교차 횟수만 세면 되므로
    닫힌 폴리곤 가정 없이 그대로 맞는다."""
    crossings = np.zeros(x.shape, dtype=np.int64)
    for a, b in lines:
        x1, y1 = p[a]
        x2, y2 = p[b]
        if x1 == x2:
            continue  # 수직 선분은 수직 반직선과 만나지 않음(측면 접촉, 측도 0)
        lo_x, hi_x = (x1, x2) if x1 < x2 else (x2, x1)
        in_x = (x >= lo_x) & (x < hi_x)  # 반개구간: 꼭짓점 중복 계산 방지
        t = (x - x1) / (x2 - x1)
        y_at = y1 + t * (y2 - y1)
        crossings += (in_x & (y_at <= y) & (y_at > y_below)).astype(np.int64)
    return crossings % 2 == 0


def _dist_to_segments(x: np.ndarray, y: np.ndarray, p: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """격자점마다 가장 가까운 선분(점이 아니라 선분 자체)까지의 최단거리.
    cKDTree(p).query는 꼭짓점까지 거리라 선분이 길고 꼭짓점이 드물면 값이 부정확해진다
    — 실측 스냅샷처럼 점이 촘촘하면 거의 같지만, 부호 있는 값 검증(해석적 입력)에는
    선분 기준 거리가 필요하다."""
    dist = np.full(x.shape, np.inf, dtype=np.float64)
    for a, b in lines:
        x1, y1 = p[a]
        x2, y2 = p[b]
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            d = np.hypot(x - x1, y - y1)
        else:
            t = np.clip(((x - x1) * dx + (y - y1) * dy) / seg_len2, 0.0, 1.0)
            d = np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        np.minimum(dist, d, out=dist)
    return dist


def signed_dist_field(
    p: np.ndarray,
    lines: np.ndarray,
    nx: int = 256,
    ny: int = 128,
    xr: tuple[float, float] = (-15, 15),
    yr: tuple[float, float] | None = None,
) -> np.ndarray:
    """부호 있는 거리장(SDF). solid(마스크+Si) 내부=음수, vacuum(식각된 공간)=양수
    [제안: 부호 규약 — 레벨셋/그래픽스에서 흔한 "내부 음수" 관례를 따름, 다른 근거 없음]."""
    if yr is None:
        yr = auto_yr()
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    x, y = gx.ravel(), gy.ravel()
    dist = _dist_to_segments(x, y, p, lines)
    inside = _inside_solid(x, y, p, lines, y_below=yr[0] - 1.0)
    sign = np.where(inside, -1.0, 1.0)
    return (sign * dist).reshape(ny, nx).astype(np.float32)


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
        np.save(out / f"sdf{k:03d}.npy", signed_dist_field(p, lines))
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
    max_workers: int | None = 1,
) -> dict:
    """splits(train/val/test/ood -> run_id 리스트)를 raw_dir에서 읽어
    out_root/<dataset_hash>/graphs/<run_id>/에 쓰고 manifest.json·splits.json을 남긴다.

    run_id마다 출력 경로가 독립적이라(공유 상태 없음) max_workers로 run_id 단위
    ProcessPoolExecutor 병렬화를 켤 수 있다. 기본값 1(직렬)은 기존 동작과 동일 —
    프로세스 풀 생성 자체에 고정 비용(Windows에서 실측 약 1초/워커)이 있어 레시피
    수가 적으면(테스트 픽스처 등) 병렬화가 오히려 손해다. 레시피가 많은 본 데이터
    생성(N=400 규모)에서만 호출자가 명시적으로 max_workers>1을 넘기는 걸 의도함.
    signed_dist_field 내부를 (E,N) 브로드캐스트로 벡터화하는 방향은 실측상
    역효과였음(W3-09 평가, reports/evidence/W3-09_etl_parallel.md) — run_id 단위
    병렬화만 적용."""
    run_ids = sorted({rid for group in splits.values() for rid in group})
    dataset_hash = compute_dataset_hash(run_ids, raw_dir)
    out = pathlib.Path(out_root) / dataset_hash
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    raw = pathlib.Path(raw_dir)

    if max_workers == 1 or len(run_ids) <= 1:
        for rid in run_ids:
            etl_run(raw / rid, out / "graphs" / rid, theta_keys)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(etl_run, raw / rid, out / "graphs" / rid, theta_keys) for rid in run_ids
            ]
            for fut in futures:
                fut.result()  # 워커 예외를 여기서 바로 전파(조용히 묻히지 않도록)

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
