"""ViennaPS 스냅샷(.vtp) → 모델 입력 SDF. 학습 데이터를 만든 수정 ETL(fixed-v2)과 같은 계산.
수정 내용: (1) 마스크/Si 경계선(표면 아님) 제거 (2) 안/밖 판정 출발점을 가장 깊은 점 아래로 (3) 격자 오른쪽 끝 열 보정
사용:
    from preprocess import vtp_to_sdf
    sdf0 = vtp_to_sdf('t000.vtp', cfg)      # (NY, NX) µm, 고체 음수 / 빈 공간 양수
필요: pyvista
"""
from collections import defaultdict
import numpy as np

def read_vtp(path):
    import pyvista as pv
    m = pv.read(str(path))
    return np.asarray(m.points)[:, :2].astype(np.float64), m.lines.reshape(-1, 3)[:, 1:]

def surface_lines(p, lines, tol=0.05):
    """마스크/실리콘 경계선(도메인 벽의 y≈0 끝점 → 세 갈래 꼭짓점까지)을 빼고 진짜 표면 선분만 남김."""
    adj = defaultdict(list)
    for e, (a, b) in enumerate(lines):
        adj[a].append(e); adj[b].append(e)
    starts = [v for v, es in adj.items() if len(es) == 1 and abs(p[v, 1]) < tol]
    is_if = np.zeros(len(lines), bool)
    for v in starts:
        walked, cur, prev_e = [], v, None
        while True:
            nxt = [e for e in adj[cur] if e != prev_e]
            if cur != v and len(adj[cur]) >= 3: break
            if not nxt or len(walked) > len(lines): walked = []; break
            e = nxt[0]; walked.append(e); a, b = lines[e]; cur = b if a == cur else a; prev_e = e
        is_if[walked] = True
    return lines[~is_if]

def vtp_to_sdf(path, cfg):
    g = cfg['grid']
    p, lines = read_vtp(path); surf = surface_lines(p, lines)
    gx, gy = np.meshgrid(np.linspace(g['x_min_um'], g['x_max_um'], g['nx']), np.linspace(g['y_bottom_um'], g['y_top_um'], g['ny']))
    x, y = gx.ravel(), gy.ravel()
    dist = np.full(x.shape, np.inf)
    for a, b in surf:
        (x1, y1), (x2, y2) = p[a], p[b]; dx, dy = x2 - x1, y2 - y1; L2 = dx * dx + dy * dy
        if L2 == 0: d = np.hypot(x - x1, y - y1)
        else:
            t = np.clip(((x - x1) * dx + (y - y1) * dy) / L2, 0, 1); d = np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        np.minimum(dist, d, out=dist)
    y_below = min(g['y_bottom_um'], float(p[:, 1].min())) - 1.0
    xi = np.minimum(x, g['x_max_um'] - 1e-6); cross = np.zeros(x.shape, np.int64)
    for a, b in surf:
        (x1, y1), (x2, y2) = p[a], p[b]
        if x1 == x2: continue
        lo, hi = min(x1, x2), max(x1, x2); y_at = y1 + (xi - x1) / (x2 - x1) * (y2 - y1)
        cross += ((xi >= lo) & (xi < hi) & (y_at <= y) & (y_at > y_below)).astype(np.int64)
    return (np.where(cross % 2 == 0, -1.0, 1.0) * dist).reshape(g['ny'], g['nx']).astype(np.float32)
