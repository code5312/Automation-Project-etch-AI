"""C4: 평가 지표(최소판) — Chamfer 거리, 깊이. 나머지(IoU, CD, 테이퍼 등)는 5주차(W5-01).

torch 없이 scipy(cKDTree)만으로 동작한다 — eval은 train 전용 패키지(torch_geometric 등)에
의존하지 않아야 서빙·CI 어디서든 가볍게 돌릴 수 있다(설계서 §6 환경 분리 취지).
"""

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(pred: np.ndarray, true: np.ndarray) -> float:
    """두 점집합 사이의 양방향 평균 최근접 거리. 점 개수가 달라도 되고 순서는 무관하다."""
    tree_true = cKDTree(true)
    tree_pred = cKDTree(pred)
    d_pred_to_true, _ = tree_true.query(pred)
    d_true_to_pred, _ = tree_pred.query(true)
    return float(d_pred_to_true.mean() + d_true_to_pred.mean())


def depth(points: np.ndarray) -> float:
    """식각 깊이: Si 윗면 y=0 기준 -ymin (etchpipe.etl.etl.metrics와 동일 정의)."""
    return float(-points[:, 1].min())


def depth_error(pred: np.ndarray, true: np.ndarray) -> float:
    return abs(depth(pred) - depth(true))
