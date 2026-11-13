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


def _restrict_to_silicon(points: np.ndarray) -> np.ndarray:
    """y≤0(실리콘 영역)인 점만 남긴다 [제안, ADR-0005 결정1]. 마스크(y>0)는 식각 중
    거의 변하지 않아서(문서 실측: 높이 손실 최대 0.013µm, 400건 전부 마스크 유지)
    포함시키면 거리 지표가 "거의 안 변하는 부분"에 묻혀 왜곡된다(문서 실측: 마스크
    포함 표면 Dice 0.59 vs 제외 0.01). 점이 하나도 안 남으면(모두 y>0) 빈 배열을
    반환 — 호출부가 처리한다.

    **비교 대상(상대 곡선)은 restrict하지 않고 그대로 둔다** [제안] — 쿼리 점만
    실리콘 영역으로 제한하고, 투영 대상 선분은 전체를 유지해야 y=0 경계 바로 아래
    점이 "끊긴 선분" 때문에 거리가 왜곡되는 걸 피한다."""
    return points[points[:, 1] <= 0]


def chamfer_distance(
    pred: np.ndarray, true: np.ndarray, densify_step: float = 0.05, silicon_only: bool = False
) -> float:
    """두 윤곽선 사이의 양방향 평균 점-선분 거리 [확인, ADR-0004 결정1 — 골든셋 공식
    채점 노트북(`golden10_채점_(입력용).ipynb`) `chamfer()`와 동일 정의로 교체, 2026-11-11].
    각 곡선을 `densify_step`(기본 0.05µm) 간격으로 촘촘히 재샘플링한 뒤 상대 곡선의
    선분까지 거리를 재서 양방향 평균(÷2)한다. 두 곡선의 원래 점 밀도가 달라도(예: MGN
    메시 vs ViennaPS 스냅샷) 왜곡되지 않는다. **주의**: `train/chamfer_loss.py`(학습
    손실)는 역전파·속도 때문에 의도적으로 점-점 방식을 그대로 쓴다 — 이 함수와 더 이상
    같은 정의가 아니다.

    `silicon_only=True`면 ADR-0005 결정1대로 y≤0만 비교한다 — **기본값은 False**(기존
    호출부인 `monitor.py` 드리프트 판정·`gate.py` ④ 롤아웃 검사의 기존 수치를 이번
    1단계에서 바꾸지 않기 위함)[제안, 2026-11-14]. 새 평가 전용 경로(표면 Dice·HD95,
    ADR-0005 2단계 게이트)에서만 명시적으로 True를 쓴다."""
    pred_d, true_d = _densify_curve(pred, densify_step), _densify_curve(true, densify_step)
    if silicon_only:
        pred_d, true_d = _restrict_to_silicon(pred_d), _restrict_to_silicon(true_d)
    d_pred_to_true = _distance_to_curve(pred_d, true)
    d_true_to_pred = _distance_to_curve(true_d, pred)
    return 0.5 * float(d_pred_to_true.mean() + d_true_to_pred.mean())


def surface_dice(
    pred: np.ndarray, true: np.ndarray, tau: float = 0.3, densify_step: float = 0.05
) -> float:
    """표면 Dice [제안, ADR-0005 결정1, 문서 §2.1 — Nikolov 외 2021]. 두 윤곽선을
    `densify_step`(0.05µm) 간격으로 조밀화한 뒤, 상대 곡선에서 `tau`(기본 0.3µm) 이내인
    점의 비율을 양방향으로 합산한다(분자: 두 방향에서 tau 이내인 점 개수 합, 분모: 두
    곡선의 전체 점 개수 합). y≤0(실리콘 영역)만 비교한다 — `_restrict_to_silicon` 참고.
    양쪽 다 점이 하나도 안 남으면(완전히 마스크만 있는 경우, 사실상 안 생길 케이스) 1.0."""
    pred_d = _restrict_to_silicon(_densify_curve(pred, densify_step))
    true_d = _restrict_to_silicon(_densify_curve(true, densify_step))
    if len(pred_d) == 0 and len(true_d) == 0:
        return 1.0
    if len(pred_d) == 0 or len(true_d) == 0:
        return 0.0
    hit_pred = (_distance_to_curve(pred_d, true) <= tau).sum()
    hit_true = (_distance_to_curve(true_d, pred) <= tau).sum()
    return float(hit_pred + hit_true) / float(len(pred_d) + len(true_d))


def hd95(pred: np.ndarray, true: np.ndarray, densify_step: float = 0.05) -> float:
    """HD95(95백분위 하우스도르프 거리) [제안, ADR-0005 결정1, 문서 §2.1 — Huttenlocher
    외 1993, Taha & Hanbury 2015 95백분위 관례]. 양방향 거리 각각의 95번째 백분위 중
    큰 값. Chamfer(평균)가 가리는 국소 큰 오차를 잡는다. y≤0(실리콘 영역)만 비교한다."""
    pred_d = _restrict_to_silicon(_densify_curve(pred, densify_step))
    true_d = _restrict_to_silicon(_densify_curve(true, densify_step))
    if len(pred_d) == 0 or len(true_d) == 0:
        return 0.0
    p95_pred = float(np.percentile(_distance_to_curve(pred_d, true), 95))
    p95_true = float(np.percentile(_distance_to_curve(true_d, pred), 95))
    return max(p95_pred, p95_true)


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


def iou(mask_pred: np.ndarray, mask_true: np.ndarray, mask_initial: np.ndarray) -> float:
    """처음 고체였다가 식각된 칸 기준 IoU [제안, ADR-0005 결정2 — 기존 정의(전체 vacuum
    영역)를 대체]. `mask_initial`은 t=0(식각 시작 전) 마스크(solid=True) — 웨이퍼 위
    공기 영역처럼 애초에 항상 vacuum인 칸은 예측·정답에서 항상 일치해 IoU를 부풀리므로
    뺀다. `mask_pred`/`mask_true`도 `mask_initial`과 같은 격자(xr/yr/nx/ny)여야 한다.
    둘 다 "처음 고체였다가 식각된 칸"이 없으면(식각 전혀 안 됨) 1.0으로 본다."""
    etched_pred = (~mask_pred) & mask_initial
    etched_true = (~mask_true) & mask_initial
    union = np.logical_or(etched_pred, etched_true).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(etched_pred, etched_true).sum()
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
    top_frac: float = 0.1,
    mid_frac: float = 0.5,
    bottom_frac: float = 0.9,
) -> dict[str, float]:
    """top/mid/bot CD — 깊이의 10%/50%/90% 높이에서 측정 [제안, ADR-0005 결정2, 문서
    §2.2 — 기존(y=0, 45%, 90%)을 대체]. y=0(마스크 구멍 폭)은 더 이상 top이 아니다 —
    0%·100% 끝단은 값이 불안정해(문서 §5 "10·50·90% CD 측정 높이" 근거) 10%에서 잰다."""
    return {
        "top": cd_at_y(mask, xr, yr, -depth_value * top_frac),
        "mid": cd_at_y(mask, xr, yr, -depth_value * mid_frac),
        "bot": cd_at_y(mask, xr, yr, -depth_value * bottom_frac),
    }


def taper_angle_deg(cd_top: float, cd_bot: float, depth_value: float) -> float:
    """측벽이 수직(0도)에서 기울어진 각도. top→bot 선형 프로파일 가정
    tan(theta) = (cd_top - cd_bot) / 2 / (0.8 × depth) [제안, ADR-0005 결정2, 문서
    §2.2 — 분모를 `depth_value`에서 `0.8 × depth_value`로 변경]. `cd_profile`의 top/bot이
    각각 깊이 10%/90% 지점(80% 구간)에서 측정되므로, 그 구간의 실제 수직 거리인
    0.8×depth로 분모를 맞춘다(기존엔 `cd_profile`이 y=0을 top으로 썼던 구버전 기준의
    분모가 그대로 남아 있었음)."""
    if depth_value <= 0:
        return 0.0
    return float(np.degrees(np.arctan((cd_top - cd_bot) / 2.0 / (0.8 * depth_value))))


def bowing(
    mask: np.ndarray,
    xr: tuple[float, float],
    yr: tuple[float, float],
    depth_value: float,
    low_frac: float = 0.05,
    high_frac: float = 0.95,
    n_samples: int = 19,
) -> float:
    """5~95% 높이 구간의 최대 CD − 5% 높이 CD(양수=바깥으로 볼록) [제안, ADR-0005 결정2,
    문서 §2.2 — 기존 `중간CD − (위+바닥)/2`(3점 비교)를 대체]. 최대 폭 위치가 항상
    중간(50%)은 아니라서, `low_frac`~`high_frac`(기본 5~95%, 5%p 간격 19개 지점)을
    직접 스캔해 최댓값을 찾는다. **시그니처가 바뀌었다** — 기존엔 미리 구한 CD 3개
    (top/mid/bot)를 받았지만, 이제 여러 높이를 새로 재야 해서 `mask`/`xr`/`yr`/
    `depth_value`를 직접 받는다(호출부 없음 확인, 2026-11-14 — 영향 없음)."""
    fracs = np.linspace(low_frac, high_frac, n_samples)
    cds = np.array([cd_at_y(mask, xr, yr, -depth_value * f) for f in fracs])
    return float(cds.max() - cds[0])


def speed_ratio(sim_time_s: float, model_time_s: float) -> float:
    """추론 속도 비 = ViennaPS 레시피당 시간 ÷ 모델 시간 [제안, ADR-0005 결정2-1, 문서
    §2.3]. 합격선 없음 — 보고용. 클수록 모델이 더 빠르다. `model_time_s`가 0 이하면
    (측정 실패 등) `inf`를 돌려 호출부가 "측정 불가"로 구분할 수 있게 한다."""
    if model_time_s <= 0:
        return float("inf")
    return float(sim_time_s / model_time_s)


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
