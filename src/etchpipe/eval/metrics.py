"""C4: 평가 지표 — Chamfer 거리, 깊이, IoU, CD(top/mid/bot), 테이퍼, 바우잉, 추론 시간(W5-01).

torch 없이 scipy(cKDTree)만으로 동작한다 — eval은 train 전용 패키지(torch_geometric 등)에
의존하지 않아야 서빙·CI 어디서든 가볍게 돌릴 수 있다(설계서 §6 환경 분리 취지). `etl.py`가
쓰는 pyvista/shapely는 train 전용이 아니라(ETL 공통) 가져다 써도 이 취지를 안 어긴다.

설계서 §3 C4 "공정 비교": 모델 A(점 집합)와 모델 B(SDF, W6-11 이후)를 **같은 래스터
마스크**로 변환해서 IoU/CD/테이퍼/바우잉을 공통 기준으로 계산한다
(`mask_from_points`/`mask_from_sdf`). 마스크 규약은 `etl.signed_dist_field`와 동일하게
solid(마스크+Si)=True.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from etchpipe.etl.etl import signed_dist_field


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


def mask_from_sdf(sdf: np.ndarray) -> np.ndarray:
    """solid=True. `etl.signed_dist_field` 부호 규약(내부 음수)을 그대로 따른다."""
    return sdf <= 0


def mask_from_points(
    p: np.ndarray,
    lines: np.ndarray,
    nx: int = 256,
    ny: int = 128,
    xr: tuple[float, float] = (-15, 15),
    yr: tuple[float, float] | None = None,
) -> np.ndarray:
    """모델 A(점 집합+연결) 출력을 모델 B와 같은 래스터 마스크로 변환. SDF를 거쳐가므로
    `etl.signed_dist_field`의 내부 경계 필터링(마스크/Si 접촉면 제거)을 그대로 상속한다."""
    sdf = signed_dist_field(p, lines, nx=nx, ny=ny, xr=xr, yr=yr)
    return mask_from_sdf(sdf)


def iou(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    """식각된(vacuum) 영역 기준 IoU [제안] — 변하는 건 트렌치 모양(vacuum)이지 solid
    바탈이 아니라서 vacuum으로 계산한다. 둘 다 vacuum이 없으면(완전 solid) 1.0으로 본다."""
    vac_pred, vac_true = ~mask_pred, ~mask_true
    union = np.logical_or(vac_pred, vac_true).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(vac_pred, vac_true).sum()
    return float(inter) / float(union)


def cd_at_y(
    mask: np.ndarray,
    xr: tuple[float, float],
    yr: tuple[float, float],
    y_target: float,
) -> float:
    """`y_target`에 가장 가까운 격자 행에서, 중심(x=0)을 포함해 바깥으로 뻗어나가는
    연속 vacuum 구간의 폭(CD) [제안]. 중심 자체가 solid(식각 안 된 상태)면 0.0."""
    ny, nx = mask.shape
    row_idx = int(np.argmin(np.abs(np.linspace(yr[0], yr[1], ny) - y_target)))
    row = mask[row_idx]
    center = int(np.argmin(np.abs(np.linspace(xr[0], xr[1], nx))))
    if row[center]:
        return 0.0
    left = center
    while left > 0 and not row[left - 1]:
        left -= 1
    right = center
    while right < nx - 1 and not row[right + 1]:
        right += 1
    cell_w = (xr[1] - xr[0]) / (nx - 1)
    return float((right - left) * cell_w)


def cd_profile(
    mask: np.ndarray,
    xr: tuple[float, float],
    yr: tuple[float, float],
    depth_value: float,
    bottom_frac: float = 0.9,
) -> dict[str, float]:
    """top(y=0, Si 표면)/mid/bot(바닥) CD [제안]. bot은 바닥 끝점(폭이 0으로 수렴)이 아니라
    `bottom_frac`(기본 90%) 지점에서 재서 끝점 불안정성을 피한다."""
    y_bot = -depth_value * bottom_frac
    return {
        "top": cd_at_y(mask, xr, yr, 0.0),
        "mid": cd_at_y(mask, xr, yr, y_bot / 2.0),
        "bot": cd_at_y(mask, xr, yr, y_bot),
    }


def taper_angle_deg(cd_top: float, cd_bot: float, depth_value: float) -> float:
    """측벽이 수직(0도)에서 기울어진 각도. top→bot 선형 프로파일 가정
    tan(theta) = (cd_top - cd_bot) / 2 / depth [제안, 정의서 7.2 "테이퍼 각도"의 구체 구현]."""
    if depth_value <= 0:
        return 0.0
    return float(np.degrees(np.arctan((cd_top - cd_bot) / 2.0 / depth_value)))


def bowing(cd_top: float, cd_mid: float, cd_bot: float) -> float:
    """중간 CD가 top/bot 선형 평균보다 얼마나 넓은지(양수=바깥으로 볼록) [제안]."""
    return float(cd_mid - (cd_top + cd_bot) / 2.0)


def measure_latency(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """`fn(*args, **kwargs)` 실행 시간(초, `time.perf_counter`)과 결과를 같이 반환."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
