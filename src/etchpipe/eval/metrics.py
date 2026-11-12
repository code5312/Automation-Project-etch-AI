"""C4: 평가 지표 — Chamfer 거리, 깊이, IoU, CD(top/mid/bot), 테이퍼, 바우잉, 추론 시간(W5-01).

torch 없이 numpy만으로 동작한다 — eval은 train 전용 패키지(torch_geometric 등)에
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

from etchpipe.etl.etl import signed_dist_field


def _densify_curve(curve: np.ndarray, step: float) -> np.ndarray:
    """선분 사이에 `step` 간격으로 점을 채운다 [확인, ADR-0004 결정1 — 골든셋 공식 채점
    노트북 `densify()` 그대로 포팅]. 원래 점이 희소해도 균일한 간격으로 다시 샘플링해,
    곡선 길이에 비례한 평균을 낼 수 있게 한다."""
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / step).astype(int))
    parts = [
        curve[i] + np.linspace(0, 1, counts[i], endpoint=False)[:, None] * (curve[i + 1] - curve[i])
        for i in range(len(lengths))
    ]
    return np.vstack(parts + [curve[-1:]])


def _distance_to_curve(points: np.ndarray, curve: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """각 점에서 `curve`(선분들)까지의 최단거리 [확인, ADR-0004 결정1 — 노트북
    `distance_to_curve()` 그대로 포팅]. 가장 가까운 점이 아니라 선분에 수직으로 투영한
    위치까지의 거리라, 선분 끝점 사이 임의 위치가 더 가까워도 정확히 잡힌다(점-점 최근접
    방식은 점이 희소하면 과대평가됨 — ADR-0003 §4가 SDF 거리 계산에서 이미 같은 이유로
    선분 방식을 택한 전례와 일치)."""
    a, b = curve[:-1], curve[1:]
    ab = b - a
    ab2 = np.where((ab**2).sum(1) > 0, (ab**2).sum(1), 1e-30)
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk, None, :]
        t = np.clip(((p - a) * ab).sum(-1) / ab2, 0.0, 1.0)
        out[s : s + chunk] = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(1)
    return out


def chamfer_distance(pred: np.ndarray, true: np.ndarray, densify_step: float = 0.05) -> float:
    """두 윤곽선 사이의 양방향 평균 점-선분 거리 [확인, ADR-0004 결정1 — 골든셋 공식
    채점 노트북(`golden10_채점_(입력용).ipynb`) `chamfer()`와 동일 정의로 교체, 2026-11-11].
    각 곡선을 `densify_step`(기본 0.05µm) 간격으로 촘촘히 재샘플링한 뒤 상대 곡선의
    선분까지 거리를 재서 양방향 평균(÷2)한다. 두 곡선의 원래 점 밀도가 달라도(예: MGN
    메시 vs ViennaPS 스냅샷) 왜곡되지 않는다. **주의**: `train/chamfer_loss.py`(학습
    손실)는 역전파·속도 때문에 의도적으로 점-점 방식을 그대로 쓴다 — 이 함수와 더 이상
    같은 정의가 아니다."""
    d_pred_to_true = _distance_to_curve(_densify_curve(pred, densify_step), true)
    d_true_to_pred = _distance_to_curve(_densify_curve(true, densify_step), pred)
    return 0.5 * float(d_pred_to_true.mean() + d_true_to_pred.mean())


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


def self_intersection_count(curve: np.ndarray, min_loop: float = 0.3) -> int:
    """C5 게이트 신설 "물리" 규칙의 자동 계산 부분 [확인, ADR-0004 결정4 — 골든셋 공식
    채점 노트북 `self_intersections()` 그대로 포팅]. 곡선이 자기 자신과 교차한 횟수 —
    실제 단면은 스스로 교차할 수 없으므로 0이어야 한다(게이트는 이 값을 직접 계산하지
    않고, 호출부가 이 함수로 구해서 `gate.run_gate(self_intersection_count=...)`에
    넘긴다). `min_loop`(기본 격자 1칸=0.3µm)보다 작은 고리는 수치 잡음으로 보고 세지
    않는다. `curve`는 점 집합이 아니라 **순서가 있는 표면선**이어야 한다(모델 A처럼
    점 구름+연결 정보뿐이면 `etl.outer_path`로 먼저 표면선만 추출)."""

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
    return int(
        sum(np.ptp(curve[s + 1 : e + 1], axis=0).max() >= min_loop for s, e in zip(i[hit], j[hit]))
    )


def measure_latency(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """`fn(*args, **kwargs)` 실행 시간(초, `time.perf_counter`)과 결과를 같이 반환."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
