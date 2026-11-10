"""W2-06/W5-01: Chamfer·depth·IoU·CD·테이퍼·바우잉·추론시간 지표 단위 테스트.
설계서 §9: 해석적으로 답을 아는 입력으로 검증."""

import time

import numpy as np

from etchpipe.eval.metrics import (
    bowing,
    cd_at_y,
    cd_profile,
    chamfer_distance,
    depth,
    depth_error,
    iou,
    mask_from_points,
    mask_from_sdf,
    measure_latency,
    taper_angle_deg,
)


def test_chamfer_zero_when_identical() -> None:
    p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    assert chamfer_distance(p, p) == 0.0


def test_chamfer_order_invariant() -> None:
    p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, -1.0]])
    q = p[::-1].copy()
    a = chamfer_distance(p, q)
    b = chamfer_distance(q, p)
    assert abs(a - b) < 1e-9


def test_chamfer_parallel_lines_known_distance() -> None:
    # 두 평행한 직선(y=0, y=d)을 조밀하게 샘플링하면 양방향 최근접 거리는 모두 d에 수렴
    x = np.linspace(0, 10, 200)
    line_a = np.stack([x, np.zeros_like(x)], axis=1)
    d = 2.5
    line_b = np.stack([x, np.full_like(x, d)], axis=1)
    cd = chamfer_distance(line_a, line_b)
    assert abs(cd - 2 * d) < 0.05  # 양방향 합이므로 2*d


def test_depth_on_rectangular_trench() -> None:
    # 사각 트렌치: 위쪽(y=0) 평평한 면 + 바닥(y=-depth) 평평한 면
    top = np.stack([np.linspace(-5, 5, 20), np.zeros(20)], axis=1)
    bottom = np.stack([np.linspace(-2, 2, 10), np.full(10, -7.0)], axis=1)
    points = np.concatenate([top, bottom])
    assert depth(points) == 7.0


def test_depth_error_zero_when_identical() -> None:
    p = np.array([[0.0, 0.0], [1.0, -3.0]])
    assert depth_error(p, p) == 0.0


def test_depth_error_detects_difference() -> None:
    pred = np.array([[0.0, 0.0], [1.0, -3.0]])
    true = np.array([[0.0, 0.0], [1.0, -5.0]])
    assert depth_error(pred, true) == 2.0


def _rect_trench_mask(
    xr: tuple[float, float] = (-10, 10),
    yr: tuple[float, float] = (-10, 10),
    nx: int = 81,
    ny: int = 81,
    half_width: float = 2.0,
    depth_value: float = 5.0,
) -> np.ndarray:
    """측벽이 수직인 사각 트렌치: 모든 깊이에서 CD가 같아야 한다(테이퍼·바우잉 0)."""
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    vacuum = (np.abs(gx) <= half_width) & (gy <= 0) & (gy >= -depth_value)
    return ~vacuum  # solid=True


def _trap_trench_mask(
    xr: tuple[float, float],
    yr: tuple[float, float],
    nx: int,
    ny: int,
    half_width_top: float,
    half_width_bot: float,
    depth_value: float,
) -> np.ndarray:
    """top→bot 선형으로 좁아지는(또는 넓어지는) 트렌치 — taper_angle_deg 검증용."""
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    frac = np.clip(-gy / depth_value, 0.0, 1.0)
    half_width = half_width_top + (half_width_bot - half_width_top) * frac
    vacuum = (np.abs(gx) <= half_width) & (gy <= 0) & (gy >= -depth_value)
    return ~vacuum


def test_mask_from_sdf_matches_sign_convention() -> None:
    sdf = np.array([[-1.0, 1.0], [0.0, -2.0]])
    assert np.array_equal(mask_from_sdf(sdf), np.array([[True, False], [True, True]]))


def test_mask_from_points_rectangular_trench_matches_direct_mask() -> None:
    # tests/unit/test_etl.py의 사각 트렌치 점·선분 구성과 동일(마스크 top y=5, 측벽 x=±2, 바닥 y=-3)
    p = np.array(
        [[-10.0, 5.0], [-2.0, 5.0], [-2.0, -3.0], [2.0, -3.0], [2.0, 5.0], [10.0, 5.0]],
        dtype=np.float32,
    )
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    mask = mask_from_points(p, lines, nx=41, ny=41, xr=(-10, 10), yr=(-10, 10))
    xs = np.linspace(-10, 10, 41)
    ys = np.linspace(-10, 10, 41)

    def at(qx: float, qy: float) -> bool:
        return bool(mask[np.argmin(np.abs(ys - qy)), np.argmin(np.abs(xs - qx))])

    assert at(0, 0) is False  # 트렌치 안 = vacuum
    assert at(0, -5) is True  # 바닥보다 아래 = solid
    assert at(-6, 2) is True  # 마스크 아래, 트렌치 밖 = solid


def test_iou_one_for_identical_masks() -> None:
    mask = _rect_trench_mask()
    assert iou(mask, mask) == 1.0


def test_iou_zero_for_disjoint_vacuum_regions() -> None:
    xr, yr, nx, ny = (-10, 10), (-10, 10), 81, 81
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    left_vacuum = (gx >= -4) & (gx <= -2) & (gy <= 0) & (gy >= -5)
    right_vacuum = (gx >= 2) & (gx <= 4) & (gy <= 0) & (gy >= -5)
    assert iou(~left_vacuum, ~right_vacuum) == 0.0


def test_iou_partial_overlap_known_ratio() -> None:
    xr, yr, nx, ny = (-10, 10), (-10, 10), 401, 81
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    band = (gy <= 0) & (gy >= -5)
    vac_a = (gx >= -4) & (gx <= 0) & band  # 폭 4
    vac_b = (gx >= -2) & (gx <= 2) & band  # 폭 4, 2만큼 겹침
    # 교집합 [-2,0]=폭2, 합집합 [-4,2]=폭6 -> IoU = 2/6
    assert abs(iou(~vac_a, ~vac_b) - 1 / 3) < 0.02


def test_cd_at_y_rectangular_trench_matches_width() -> None:
    mask = _rect_trench_mask(half_width=2.0, depth_value=5.0, nx=161, ny=161)
    assert abs(cd_at_y(mask, (-10, 10), (-10, 10), 0.0) - 4.0) < 0.2
    assert abs(cd_at_y(mask, (-10, 10), (-10, 10), -2.5) - 4.0) < 0.2


def test_cd_at_y_zero_when_center_is_solid() -> None:
    mask = _rect_trench_mask(half_width=2.0, depth_value=5.0)
    assert cd_at_y(mask, (-10, 10), (-10, 10), -8.0) == 0.0  # 트렌치 바닥보다 아래


def test_cd_profile_rectangular_trench_top_mid_bot_equal() -> None:
    mask = _rect_trench_mask(half_width=2.0, depth_value=5.0, nx=161, ny=161)
    prof = cd_profile(mask, (-10, 10), (-10, 10), depth_value=5.0)
    assert set(prof) == {"top", "mid", "bot"}
    assert abs(prof["top"] - prof["bot"]) < 0.3
    assert abs(prof["top"] - prof["mid"]) < 0.3


def test_cd_profile_trapezoid_top_wider_than_bot() -> None:
    mask = _trap_trench_mask(
        (-10, 10), (-10, 10), 161, 161, half_width_top=4.0, half_width_bot=1.0, depth_value=5.0
    )
    prof = cd_profile(mask, (-10, 10), (-10, 10), depth_value=5.0)
    assert prof["top"] > prof["mid"] > prof["bot"]


def test_taper_angle_zero_for_vertical_sidewall() -> None:
    assert taper_angle_deg(cd_top=4.0, cd_bot=4.0, depth_value=5.0) == 0.0


def test_taper_angle_matches_known_arctan() -> None:
    angle = taper_angle_deg(cd_top=10.0, cd_bot=6.0, depth_value=10.0)
    assert abs(angle - np.degrees(np.arctan(0.2))) < 1e-9


def test_bowing_zero_when_mid_is_linear_average() -> None:
    assert bowing(cd_top=10.0, cd_mid=8.0, cd_bot=6.0) == 0.0


def test_bowing_positive_when_mid_bulges_outward() -> None:
    assert bowing(cd_top=10.0, cd_mid=12.0, cd_bot=10.0) == 2.0


def test_measure_latency_reports_elapsed_time_and_result() -> None:
    def slow(x: int) -> int:
        time.sleep(0.02)
        return x * 2

    result, elapsed = measure_latency(slow, 3)
    assert result == 6
    assert elapsed >= 0.02
