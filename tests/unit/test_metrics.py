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
    hd95,
    iou,
    mask_from_points,
    mask_from_sdf,
    measure_latency,
    self_intersection_count,
    speed_ratio,
    surface_dice,
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
    # 두 평행한 직선(y=0, y=d)을 조밀하게 샘플링하면 양방향 평균 거리는 d에 수렴
    x = np.linspace(0, 10, 200)
    line_a = np.stack([x, np.zeros_like(x)], axis=1)
    d = 2.5
    line_b = np.stack([x, np.full_like(x, d)], axis=1)
    cd = chamfer_distance(line_a, line_b)
    assert abs(cd - d) < 0.05  # ADR-0004 결정1: 양방향 평균(÷2)이므로 d


def test_chamfer_sparse_points_use_segment_not_vertex_distance() -> None:
    # ADR-0004 결정1: 점-점(최근접 점) 방식이면 점이 희소한 쪽(여기선 끝점 2개뿐)에서
    # 중간 지점이 먼 끝점까지의 대각선 거리로 과대평가된다(약 5.59, d가 아님). 점-선분
    # 방식은 점이 2개뿐이어도 선분 위 수직 거리로 재서 d에 정확히 수렴해야 한다.
    sparse = np.array([[0.0, 0.0], [10.0, 0.0]])  # 끝점 2개뿐인 직선
    x = np.linspace(0, 10, 200)
    d = 2.5
    dense = np.stack([x, np.full_like(x, d)], axis=1)
    cd = chamfer_distance(sparse, dense)
    assert abs(cd - d) < 0.05


def test_chamfer_silicon_only_ignores_points_above_y0() -> None:
    # 기본값(silicon_only=False)은 기존 수치를 그대로 유지해야 한다(monitor.py/gate.py
    # 기존 호출부 보호, ADR-0005 결정1).
    x = np.linspace(0, 10, 200)
    pred = np.stack([x, np.zeros_like(x)], axis=1)  # y=0(마스크 경계)
    true_silicon = np.stack([x, np.full_like(x, -2.5)], axis=1)  # y=-2.5(실리콘)
    true_mixed = np.concatenate([true_silicon, np.stack([x, np.full_like(x, 50.0)], axis=1)])
    cd_default = chamfer_distance(pred, true_mixed, silicon_only=False)
    cd_silicon = chamfer_distance(pred, true_mixed, silicon_only=True)
    assert cd_silicon < cd_default  # y=50(마스크 위 공기)에 끌려가지 않아야 더 작음


def test_surface_dice_one_for_identical_curves() -> None:
    x = np.linspace(0, 10, 50)
    curve = np.stack([x, -np.abs(np.sin(x))], axis=1)  # 전부 y<=0
    assert surface_dice(curve, curve) == 1.0


def test_surface_dice_within_tau_is_full_credit() -> None:
    x = np.linspace(0, 10, 200)
    pred = np.stack([x, np.full_like(x, -2.0)], axis=1)
    true = np.stack([x, np.full_like(x, -2.0 - 0.1)], axis=1)  # tau=0.3보다 가까움
    assert surface_dice(pred, true, tau=0.3) == 1.0


def test_surface_dice_far_apart_is_zero() -> None:
    x = np.linspace(0, 10, 200)
    pred = np.stack([x, np.full_like(x, -2.0)], axis=1)
    true = np.stack([x, np.full_like(x, -10.0)], axis=1)  # tau=0.3보다 훨씬 멂
    assert surface_dice(pred, true, tau=0.3) == 0.0


def test_hd95_zero_when_identical() -> None:
    x = np.linspace(0, 10, 50)
    curve = np.stack([x, -np.abs(np.sin(x))], axis=1)
    assert hd95(curve, curve) < 1e-9  # 부동소수점 오차 허용


def test_hd95_matches_known_parallel_line_distance() -> None:
    x = np.linspace(0, 10, 200)
    d = 2.5
    pred = np.stack([x, np.zeros_like(x)], axis=1)
    true = np.stack([x, np.full_like(x, -d)], axis=1)
    assert abs(hd95(pred, true) - d) < 0.05


def test_hd95_at_least_chamfer_mean_for_skewed_errors() -> None:
    # 끝 쪽 20%만 멀고 나머지 80%는 거의 일치하는 비대칭 오차 분포 — 평균(Chamfer)보다
    # 95백분위(HD95)가 커야 한다(문서 §2.1 취지: Chamfer가 가리는 국소 큰 오차를 HD95가
    # 더 크게 반영). 밀도를 step(0.05)보다 촘촘히 깔아 densify 재샘플링 왜곡을 피한다.
    x_near = np.linspace(0, 8, 1600)  # 80%: 거의 일치(조밀, dx=0.005 < step)
    x_far = np.linspace(8, 10, 400)  # 20%: 많이 벗어남(조밀, dx=0.005 < step)
    x = np.concatenate([x_near, x_far])
    pred_y = np.concatenate([np.full_like(x_near, -2.0), np.full_like(x_far, -5.0)])
    pred = np.stack([x, pred_y], axis=1)
    true = np.stack([x, np.full_like(x, -2.0)], axis=1)
    c = chamfer_distance(pred, true, silicon_only=True)
    h = hd95(pred, true)
    assert h > c
    assert abs(h - 3.0) < 0.2  # 95백분위는 멀리 떨어진 20% 쪽 값(|{-5}-{-2}|=3) 근방


def test_speed_ratio_basic() -> None:
    assert speed_ratio(sim_time_s=940.0, model_time_s=20.0) == 47.0


def test_speed_ratio_zero_model_time_is_inf() -> None:
    assert speed_ratio(sim_time_s=940.0, model_time_s=0.0) == float("inf")


def test_self_intersection_count_zero_for_simple_curve() -> None:
    curve = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    assert self_intersection_count(curve) == 0


def test_self_intersection_count_detects_crossing() -> None:
    # 나비 모양: (0,0)->(2,2)->(2,0)->(0,2) 는 중간에서 스스로 교차한다.
    curve = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    assert self_intersection_count(curve, min_loop=0.01) > 0


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
    all_solid = np.ones_like(mask)  # t=0: 아직 아무것도 식각 안 된 상태
    assert iou(mask, mask, all_solid) == 1.0


def test_iou_zero_for_disjoint_vacuum_regions() -> None:
    xr, yr, nx, ny = (-10, 10), (-10, 10), 81, 81
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    left_vacuum = (gx >= -4) & (gx <= -2) & (gy <= 0) & (gy >= -5)
    right_vacuum = (gx >= 2) & (gx <= 4) & (gy <= 0) & (gy >= -5)
    all_solid = np.ones_like(left_vacuum)
    assert iou(~left_vacuum, ~right_vacuum, all_solid) == 0.0


def test_iou_partial_overlap_known_ratio() -> None:
    xr, yr, nx, ny = (-10, 10), (-10, 10), 401, 81
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    band = (gy <= 0) & (gy >= -5)
    vac_a = (gx >= -4) & (gx <= 0) & band  # 폭 4
    vac_b = (gx >= -2) & (gx <= 2) & band  # 폭 4, 2만큼 겹침
    all_solid = np.ones_like(vac_a)
    # 교집합 [-2,0]=폭2, 합집합 [-4,2]=폭6 -> IoU = 2/6
    assert abs(iou(~vac_a, ~vac_b, all_solid) - 1 / 3) < 0.02


def test_iou_ignores_always_vacuum_air_region() -> None:
    # ADR-0005 결정2: 웨이퍼 위 공기처럼 처음부터(t=0) vacuum이던 칸은 예측·정답에서
    # 항상 일치해도 IoU 분자·분모에서 빠져야 한다 — 기존(전체 vacuum 비교) 방식이면
    # 이 "항상 일치하는 공기"가 섞여 IoU가 부풀려졌을 자리.
    xr, yr, nx, ny = (-10, 10), (-10, 10), 81, 81
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    air = gy > 5.0  # t=0에도 이미 vacuum인 영역(웨이퍼 위 공기)
    initial_solid = ~air  # 그 외는 t=0에 전부 solid
    left_vacuum = (gx >= -4) & (gx <= -2) & (gy <= 0) & (gy >= -5)
    right_vacuum = (gx >= 2) & (gx <= 4) & (gy <= 0) & (gy >= -5)
    # 공기 영역을 뺀 겹치지 않는 식각 결과는 기존과 같은 비율이어야 한다(공기가 안 섞임)
    base = iou(~left_vacuum, ~right_vacuum, initial_solid)
    # air를 양쪽 다 vacuum으로 더 채워도(실제론 항상 그럴 것) 결과가 바뀌면 안 된다
    inflated_pred = left_vacuum | air  # solid=False(vacuum)인 곳 = ~(~left_vacuum & ~air)
    inflated_true = right_vacuum | air
    assert iou(~inflated_pred, ~inflated_true, initial_solid) == base == 0.0


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
    # ADR-0005 결정2: 분모가 depth -> 0.8*depth로 바뀜
    angle = taper_angle_deg(cd_top=10.0, cd_bot=6.0, depth_value=10.0)
    assert abs(angle - np.degrees(np.arctan((10.0 - 6.0) / 2.0 / 8.0))) < 1e-9


def _bulge_trench_mask(
    xr: tuple[float, float],
    yr: tuple[float, float],
    nx: int,
    ny: int,
    half_width_ends: float,
    half_width_mid: float,
    depth_value: float,
) -> np.ndarray:
    """양 끝보다 중간이 더 넓게 볼록한 트렌치 — bowing 양수 검증용."""
    gx, gy = np.meshgrid(np.linspace(*xr, nx), np.linspace(*yr, ny))
    frac = np.clip(-gy / depth_value, 0.0, 1.0)
    half_width = half_width_ends + (half_width_mid - half_width_ends) * np.sin(np.pi * frac)
    vacuum = (np.abs(gx) <= half_width) & (gy <= 0) & (gy >= -depth_value)
    return ~vacuum


def test_bowing_zero_for_rectangular_trench() -> None:
    # ADR-0005 결정2: 시그니처가 (cd_top, cd_mid, cd_bot) -> (mask, xr, yr, depth_value)로
    # 바뀜 — 측벽이 수직인 트렌치는 5~95% 구간 어디서 재도 CD가 같아 바우잉이 0이어야 함.
    mask = _rect_trench_mask(half_width=2.0, depth_value=5.0, nx=161, ny=161)
    assert abs(bowing(mask, (-10, 10), (-10, 10), depth_value=5.0)) < 0.3


def test_bowing_positive_for_bulging_trench() -> None:
    mask = _bulge_trench_mask(
        (-10, 10), (-10, 10), 161, 161, half_width_ends=2.0, half_width_mid=4.0, depth_value=5.0
    )
    assert bowing(mask, (-10, 10), (-10, 10), depth_value=5.0) > 1.0


def test_measure_latency_reports_elapsed_time_and_result() -> None:
    def slow(x: int) -> int:
        time.sleep(0.02)
        return x * 2

    result, elapsed = measure_latency(slow, 3)
    assert result == 6
    assert elapsed >= 0.02
