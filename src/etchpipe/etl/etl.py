"""C2: raw(.vtp) -> processed(그래프 npz + 거리장 npz + 지표 + dataset_hash/manifest).

Curator 래핑 전의 순수 파이썬 버전. SDF(부호 있는 거리장)는 W3-03에서 완성.
"""

import hashlib
import json
import pathlib
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyvista as pv
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

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
    """거리장 y 격자 범위. 바닥은 depth_max+mask_height_max만큼, 위는 mask_height_max+margin만큼
    여유를 둔다 — GPU 팀원 FNO 초안(`runs/fno_draft/best.mdlus`)이 학습에 쓴 범위
    (−68~21, `docs/project-checklist.md` W4 역할분담 메모 2026-11-09)에 맞춤. 이전 버전은
    바닥도 margin만 더해 −53~21이었음(ADR-0003 최초 값) — 이 값으로 교체하면서
    `data/processed/<hash>/sdf/`(이전 범위로 생성됨)는 재생성이 필요하다
    [미확인: 재생성 일정, graphs/는 이 함수를 쓰지 않아 영향 없음]."""
    return (-(depth_max + mask_height_max), mask_height_max + margin)


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


def _filter_internal_interfaces(p: np.ndarray, lines: np.ndarray, y_below: float) -> np.ndarray:
    """addInterfaces=True로 내보낸 선분 중, 양쪽이 모두 solid인 내부 경계(예: 마스크가
    덮어서 아직 vacuum에 노출되지 않은 Si 상단, mask/Si 접촉면)를 제거하고 실제
    solid-vacuum 외곽 경계만 남긴다.

    버그 발견 경위: MaterialIds만으로는 내부 경계와 외곽 경계를 구분할 수 없음을
    실측으로 확인함(마스크/Si 내부 접촉면과, 트렌치 바닥에서 Si가 실제로 vacuum과
    맞닿는 외곽 경계가 둘 다 MaterialIds=10으로 동일). 내부 경계를 그대로 두면
    `_inside_solid`의 짝수-홀수 교차 판정이 깨져 마스크 내부가 vacuum으로 잘못
    나온다(실측: 마스크 몸통 안쪽 점이 inside_solid=False, sdf>0으로 나왔음).

    해결: 열린 체인(도메인 좌우 경계에서 끝나는 미완성 경계, ADR-0003)을 y_below
    (항상 solid로 간주하는 바닥)까지 내려 임시로 닫은 뒤 polygonize+union하면,
    서로 다른 두 폴리곤이 공유하는 내부 경계는 union 결과의 경계(boundary)에서
    사라지고 내부로 들어간다 — 그 경계 위에 남아있는 원래 선분만 돌려준다.
    (임시로 덧붙인 닫는 선분 자체는 반환하지 않음 — `_dist_to_segments`의 거리
    계산을 오염시키지 않기 위해, 원래 있던 선분만 걸러서 넘겨야 한다.)"""
    deg: dict[int, int] = {}
    for a, b in lines:
        deg[int(a)] = deg.get(int(a), 0) + 1
        deg[int(b)] = deg.get(int(b), 0) + 1
    ends = [i for i, d in deg.items() if d == 1]

    segs = [LineString([p[a], p[b]]) for a, b in lines]
    closing: list[LineString] = [LineString([p[i], (float(p[i, 0]), y_below)]) for i in ends]
    uniq_x = sorted({round(float(p[i, 0]), 6) for i in ends})
    for i in range(len(uniq_x) - 1):
        closing.append(LineString([(uniq_x[i], y_below), (uniq_x[i + 1], y_below)]))

    # polygonize는 선분들이 교차점에서 깨끗이 노드되어 있어야 한다. 열린 끝점이 여러
    # 개(도메인 좌우 경계에 Si쪽·마스크쪽 끝이 따로 있는 경우)면 같은 x로 내려가는
    # 수직 닫는 선분들이 서로 겹치는(collinear overlap) 구간이 생기는데, 이걸 노드
    # 처리 안 하면 polygonize가 다각형을 하나도 못 찾는다(실측으로 확인) — unary_union로
    # 먼저 노드시킨다.
    noded = unary_union(segs + closing)
    polys = list(polygonize(noded))
    if not polys:
        return lines  # 닫힌 영역을 못 만드는 이상 케이스 — 안전하게 원래 선분 그대로 둔다

    boundary = unary_union(polys).boundary
    mids = (p[lines[:, 0]] + p[lines[:, 1]]) / 2.0
    mid_pts = shapely.points(mids[:, 0], mids[:, 1])
    on_boundary = shapely.dwithin(boundary, mid_pts, 1e-6)
    return lines[on_boundary]


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
    y_below = yr[0] - 1.0
    boundary_lines = _filter_internal_interfaces(p, lines, y_below)
    dist = _dist_to_segments(x, y, p, boundary_lines)
    inside = _inside_solid(x, y, p, boundary_lines, y_below=y_below)
    sign = np.where(inside, -1.0, 1.0)
    return (sign * dist).reshape(ny, nx).astype(np.float32)


def outer_path(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """플라즈마에 닿는 바깥 표면선만 추출 [확인, ADR-0004 — 골든셋 공식 채점 노트북
    `outer_path()` 그대로 포팅]. `load_snapshot`의 `lines`는 마스크/Si 내부 경계까지
    섞인 전체 메시 연결이라(`_filter_internal_interfaces`가 SDF 계산에서만 이를 걸러내는
    것과 같은 이유), 왼쪽 위 열린 끝점(x<0, y 최대)에서 오른쪽 위 열린 끝점(x>0, y
    최대)까지의 경로만 남겨야 표면만 남는다. 열린 끝점은 인접 선분이 1개뿐인 점."""
    neighbors: list[list[int]] = [[] for _ in range(len(points))]
    for a, b in lines:
        neighbors[a].append(b)
        neighbors[b].append(a)
    ends = [i for i in range(len(points)) if len(neighbors[i]) == 1]
    start = max((i for i in ends if points[i, 0] < 0), key=lambda i: points[i, 1])
    goal = max((i for i in ends if points[i, 0] > 0), key=lambda i: points[i, 1])
    previous: dict[int, int | None] = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in neighbors[u]:
            if v not in previous:
                previous[v] = u
                queue.append(v)
    path = [goal]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    return points[np.array(path[::-1])]


def _first_self_crossing(curve: np.ndarray) -> tuple[int, int] | None:
    """`curve`(순서 있는 표면선)에서 서로 가로지르는, 인접하지 않은 첫 선분 쌍(i, i+1<j).
    없으면 `None` [확인, mgn_tuned `inference.py`의 `first_cross` 그대로 포팅 — `eval.
    metrics.self_intersection_count`와 같은 교차 판정식(side)을 재사용]."""

    def side(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.sign(
            (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
            - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        )

    a, b = curve[:-1], curve[1:]
    i, j = np.triu_indices(len(a), k=2)
    hit = (side(a[i], b[i], a[j]) * side(a[i], b[i], b[j]) < 0) & (
        side(a[j], b[j], a[i]) * side(a[j], b[j], b[i]) < 0
    )
    if not hit.any():
        return None
    pair = min(zip(i[hit].tolist(), j[hit].tolist()))
    return pair


def remove_self_crossing_loops(curve: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """자기교차 고리를 교차점 하나로 바꾼다(모델 출력 후처리, 정답은 안 씀) [확인,
    ADR-0004 결정4 배선(2026-11-11) — mgn_tuned `inference.py`의 `remove_loops` 그대로
    포팅]. 정답 비교에는 안 쓰고, 모델 예측 표면선을 채점 직전에만 다듬는다."""
    for _ in range(max_iter):
        hit = _first_self_crossing(curve)
        if hit is None:
            break
        i, j = hit
        p, r = curve[i], curve[i + 1] - curve[i]
        q, s = curve[j], curve[j + 1] - curve[j]
        den = r[0] * s[1] - r[1] * s[0]
        t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
        curve = np.vstack([curve[: i + 1], (p + t * r)[None], curve[j + 1 :]])
    return curve


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


def compute_stats(graphs_root: str | pathlib.Path, run_ids: list[str]) -> dict:
    """run_ids(설계서 FR-05: train에서만 호출)의 그래프 npz(node/edge)·SDF npy를 모아
    정규화에 쓸 채널별 평균/표준편차를 계산한다. 분산은 E[x^2]-E[x]^2로 1-pass 누적
    (run_id 전체를 메모리에 올리지 않기 위함), 부동소수 오차로 음수가 나올 수 있어
    `np.maximum(var, 0)`로 클램프."""
    root = pathlib.Path(graphs_root)
    node_sum = node_sq = edge_sum = edge_sq = None
    node_n = edge_n = 0
    sdf_sum = sdf_sq = 0.0
    sdf_n = 0
    for rid in sorted(run_ids):
        d = root / rid
        for f in sorted(d.glob("g*.npz")):
            z = np.load(f)
            node = z["node"].astype(np.float64)
            edge = z["edge"].astype(np.float64)
            if node_sum is None:
                node_sum, node_sq = node.sum(0), (node**2).sum(0)
                edge_sum, edge_sq = edge.sum(0), (edge**2).sum(0)
            else:
                node_sum += node.sum(0)
                node_sq += (node**2).sum(0)
                edge_sum += edge.sum(0)
                edge_sq += (edge**2).sum(0)
            node_n += len(node)
            edge_n += len(edge)
        for f in sorted(d.glob("sdf*.npy")):
            s = np.load(f).astype(np.float64)
            sdf_sum += float(s.sum())
            sdf_sq += float((s**2).sum())
            sdf_n += s.size

    node_mean = node_sum / node_n
    node_std = np.sqrt(np.maximum(node_sq / node_n - node_mean**2, 0.0))
    edge_mean = edge_sum / edge_n
    edge_std = np.sqrt(np.maximum(edge_sq / edge_n - edge_mean**2, 0.0))
    sdf_mean = sdf_sum / sdf_n
    sdf_std = float(np.sqrt(max(sdf_sq / sdf_n - sdf_mean**2, 0.0)))

    return {
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
        "sdf_mean": float(sdf_mean),
        "sdf_std": sdf_std,
        "n_nodes": int(node_n),
        "n_edges": int(edge_n),
        "n_sdf_values": int(sdf_n),
    }


def process_dataset(
    raw_dir: str | pathlib.Path,
    splits: dict[str, list[str]],
    out_root: str | pathlib.Path,
    theta_keys: list[str],
    max_workers: int | None = 1,
) -> dict:
    """splits(train/val/test/ood -> run_id 리스트)를 raw_dir에서 읽어
    out_root/<dataset_hash>/graphs/<run_id>/에 쓰고 manifest.json·splits.json·stats.json을
    남긴다. stats.json은 FR-05(설계서 C2)대로 train split에서만 계산한다.

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
    stats = compute_stats(out / "graphs", splits["train"])  # FR-05: train에서만
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    manifest = {
        "dataset_hash": dataset_hash,
        "theta_keys": theta_keys,
        "n_recipes": len(run_ids),
        "splits_sizes": {k: len(v) for k, v in splits.items()},
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    return manifest


def validate_run(run_dir: str | pathlib.Path) -> dict:
    """W3-05: raw run 1건 검증. 예외를 던지지 않고 errors 리스트로 모아 반환한다
    (배치 검증에서 한 건이 전체를 중단시키지 않도록).
    - 스냅샷 개수 == actual_k+1(meta.json, 조기종료면 K보다 작을 수 있음)
    - 점 수>0, 점·법선에 NaN/Inf 없음(`np.isfinite`)
    - 식각 깊이가 시간에 따라 감소하지 않음(물리 타당성, 설계서 §9)
    - 점 수 최대치(그래프 크기 예산, 설계서 §4.2 위험)도 함께 보고

    **W7-03 실패 시나리오로 발견·수정**(2026-11-14): `meta.json`이 없거나(시뮬레이션이
    중간에 죽어 파일이 아예 안 쓰인 경우) 스냅샷 `.vtp`가 손상된 경우(예: 쓰기 도중
    끊김) 전부 이 함수의 "예외 안 던짐" 약속을 어기고 그대로 크래시했다(`meta.json`
    없으면 `FileNotFoundError`, 손상된 `.vtp`는 `load_snapshot` 내부에서
    `KeyError: 'MaterialIds'` 등 VTK 파싱 실패가 그대로 전파) — `validate_dataset`로
    여러 run을 배치 검증할 때 run 1개의 손상이 전체 배치를 중단시켰다. 둘 다 잡아서
    errors에 담는다."""
    run = pathlib.Path(run_dir)
    errors: list[str] = []
    try:
        meta = json.loads((run / "meta.json").read_text())
        expected = meta["actual_k"] + 1
    except (OSError, json.JSONDecodeError, KeyError) as e:
        return {
            "run_id": run.name,
            "errors": [f"meta.json unreadable: {type(e).__name__}: {e}"],
            "n_snapshots": 0,
            "max_n_points": 0,
        }
    snaps = sorted((run / "snapshots").glob("t*.vtp"))
    if len(snaps) != expected:
        errors.append(f"snapshot_count={len(snaps)} expected={expected}(actual_k+1)")

    depths = []
    max_n_points = 0
    for s in snaps:
        try:
            p, _lines, _mid, nv = load_snapshot(s)
        except Exception as e:  # noqa: BLE001 - 손상된 VTK 파일이 무엇을 던질지 보장 없음
            errors.append(f"{s.name}: unreadable snapshot: {type(e).__name__}: {e}")
            continue
        if len(p) == 0:
            errors.append(f"{s.name}: n_points=0")
        if not (np.isfinite(p).all() and np.isfinite(nv).all()):
            errors.append(f"{s.name}: NaN/Inf in points or normals")
        max_n_points = max(max_n_points, len(p))
        depths.append(metrics(p)["depth"])

    if depths and not all(b >= a - 1e-6 for a, b in zip(depths, depths[1:])):
        errors.append(f"depth not monotonically nondecreasing: {depths}")

    return {
        "run_id": run.name,
        "errors": errors,
        "n_snapshots": len(snaps),
        "max_n_points": max_n_points,
    }


def validate_dataset(raw_dir: str | pathlib.Path, run_ids: list[str] | None = None) -> dict:
    """raw_dir 아래 run_ids(생략 시 meta.json이 있는 전체)를 validate_run으로 검사해
    집계한다. 점 수 최대치(그래프 크기 예산)는 전체 run 중 최댓값."""
    raw = pathlib.Path(raw_dir)
    if run_ids is None:
        run_ids = sorted(d.name for d in raw.iterdir() if d.is_dir() and (d / "meta.json").exists())
    results = [validate_run(raw / rid) for rid in run_ids]
    bad = [r for r in results if r["errors"]]
    return {
        "n_runs": len(results),
        "n_bad": len(bad),
        "max_n_points": max((r["max_n_points"] for r in results), default=0),
        "bad_runs": bad,
    }


THETA_KEYS = [
    "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
    "ion_exponent", "mask_width", "mask_height",
]

if __name__ == "__main__":
    rid = sys.argv[1]
    rows = etl_run(f"data/raw/{rid}", f"data/processed/{rid}", THETA_KEYS)
    print(rows[0], "...", rows[-1])
