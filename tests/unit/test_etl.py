"""W2-04: ETL(그래프 변환, dataset_hash, manifest.json) 단위 테스트."""

import json
from pathlib import Path

import numpy as np
import pytest

from etchpipe.etl.etl import (
    DEPTH_MAX,
    MASK_HEIGHT_MAX_OOD,
    THETA_KEYS,
    auto_yr,
    compute_dataset_hash,
    compute_stats,
    dist_field,
    etl_run,
    metrics,
    process_dataset,
    signed_dist_field,
    to_graph,
    validate_dataset,
    validate_run,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
FIXTURE_RUNS = ["r0000", "r0001", "r0002"]


# --- 해석적으로 답을 아는 입력 (설계서 §9) ---------------------------------


def test_to_graph_edge_feat_is_relative_position_and_norm() -> None:
    p = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)  # 거리 5인 두 점
    lines = np.array([[0, 1]])
    mid = np.array([0.0, 0.0])
    nv = np.array([[0.0, 1.0], [0.0, 1.0]])
    theta = np.zeros(7, dtype=np.float32)
    nf, ei, ef = to_graph(p, lines, mid, nv, theta, t=0.0)

    assert nf.shape == (2, 14)
    assert ei.shape == (2, 2)  # 양방향이라 엣지 2개
    # 0->1 방향 엣지: rel=(3,4), norm=5
    idx01 = next(i for i in range(ei.shape[1]) if ei[0, i] == 0 and ei[1, i] == 1)
    np.testing.assert_allclose(ef[idx01], [3.0, 4.0, 5.0], atol=1e-5)


def test_to_graph_material_onehot_matches_w1_s5_mapping() -> None:
    # W1-S5 확정: MaterialIds 0=mask, 10=Si. mat1h 컬럼 [0]=mask(<5), [1]=Si(>=5).
    p = np.zeros((2, 2), dtype=np.float32)
    lines = np.array([[0, 1]])
    mid = np.array([0.0, 10.0])  # 점0=mask, 점1=Si
    nv = np.zeros((2, 2), dtype=np.float32)
    theta = np.zeros(7, dtype=np.float32)
    nf, _, _ = to_graph(p, lines, mid, nv, theta, t=0.0)
    mat1h = nf[:, 4:6]
    assert mat1h[0].tolist() == [1.0, 0.0]  # mask
    assert mat1h[1].tolist() == [0.0, 1.0]  # Si


def test_dist_field_on_single_point() -> None:
    p = np.array([[0.0, 0.0]], dtype=np.float32)
    d = dist_field(p, nx=4, ny=4, xr=(-1, 1), yr=(-1, 1))
    # 격자점 중 (0,0)에 가장 가까운 칸의 거리는 grid 간격의 반 이하여야 한다
    assert d.min() < 1.0
    assert d.shape == (4, 4)


def test_metrics_depth_is_negative_of_min_y() -> None:
    p = np.array([[0.0, 2.0], [1.0, -5.0]], dtype=np.float32)
    m = metrics(p)
    assert m["depth"] == 5.0
    assert m["ymin"] == -5.0


# --- W3-03: 부호 있는 SDF / y 범위 자동 결정 ---------------------------------


def test_auto_yr_covers_depth_max_and_mask_height_with_margin() -> None:
    lo, hi = auto_yr()
    assert lo <= -DEPTH_MAX
    assert hi >= MASK_HEIGHT_MAX_OOD
    assert lo < 0 < hi


def test_signed_dist_field_sign_and_value_straight_line() -> None:
    # 수평 직선(y=0): 아래(solid)=음수, 위(vacuum)=양수, 값은 수직 거리와 일치
    p = np.array([[-10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    lines = np.array([[0, 1]])
    sdf = signed_dist_field(p, lines, nx=21, ny=21, xr=(-5, 5), yr=(-5, 5))
    assert np.all(sdf[0] < 0)  # y=-5 행: solid
    assert np.all(sdf[-1] > 0)  # y=5 행: vacuum
    np.testing.assert_allclose(np.abs(sdf[0]), 5.0, atol=0.3)
    np.testing.assert_allclose(np.abs(sdf[-1]), 5.0, atol=0.3)


def test_signed_dist_field_rectangular_trench_sign_and_value() -> None:
    # 사각 트렌치: 마스크 top y=5(폭 밖), 측벽 x=±2, 바닥 y=-3(폭 안)
    p = np.array(
        [[-10.0, 5.0], [-2.0, 5.0], [-2.0, -3.0], [2.0, -3.0], [2.0, 5.0], [10.0, 5.0]],
        dtype=np.float32,
    )
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    sdf = signed_dist_field(p, lines, nx=41, ny=41, xr=(-10, 10), yr=(-10, 10))
    gx = np.linspace(-10, 10, 41)
    gy = np.linspace(-10, 10, 41)

    def val(qx: float, qy: float) -> float:
        return float(sdf[np.argmin(np.abs(gy - qy)), np.argmin(np.abs(gx - qx))])

    assert val(0, 0) > 0  # 트렌치 안(바닥 위, 식각된 빈 공간) = vacuum
    assert val(0, -5) < 0  # 트렌치 바닥보다 아래 = Si 벌크 = solid
    assert val(-6, 2) < 0  # 트렌치 바깥, 마스크 아래 = solid
    assert val(-6, 7) > 0  # 마스크 위 = vacuum
    np.testing.assert_allclose(abs(val(0, -3)), 0.0, atol=0.6)  # 바닥 경계 위는 값~0


def test_signed_dist_field_bowing_void_inside_outside() -> None:
    """보잉(언더컷) 프로파일 단순화: solid 속에 완전히 둘러싸인 식각 공극(void) 하나.
    한 x에서 경계를 2번 지나는(비단조) 프로파일에서도 even-odd 규칙이 맞는지 확인."""
    n = 24
    # 축에 꼭짓점이 안 걸리게 반칸 회전
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    cx, cy, r = 0.0, -5.0, 1.5
    p = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1).astype(np.float32)
    lines = np.array([[i, (i + 1) % n] for i in range(n)])
    sdf = signed_dist_field(p, lines, nx=31, ny=31, xr=(-3, 3), yr=(-8, -2))
    gx = np.linspace(-3, 3, 31)
    gy = np.linspace(-8, -2, 31)

    def val(qx: float, qy: float) -> float:
        return float(sdf[np.argmin(np.abs(gy - qy)), np.argmin(np.abs(gx - qx))])

    assert val(0.0, -5.0) > 0  # 공극 중심 = vacuum
    assert val(0.0, -2.0) < 0  # 공극 위쪽 solid(원 바깥, 2번째 교차 지남)
    assert val(0.0, -8.0) < 0  # 공극 아래쪽 solid(원 바깥, 교차 전)


def test_signed_dist_field_excludes_internal_material_interface() -> None:
    """실측 버그(2026-11-06, r0001 t000.vtp에서 재현): ViennaPS가 addInterfaces=True로
    내보내는 메시에는 solid-vacuum 외곽 경계뿐 아니라 solid-solid 내부 경계(마스크가
    덮어서 아직 vacuum에 노출되지 않은 Si 상단)도 같은 선분 배열에 MaterialIds 구분 없이
    섞여 있다. 이 내부 경계를 그대로 두면 짝수-홀수 교차 판정이 깨져 마스크 몸통
    내부가 vacuum으로 잘못 나왔다. 실제 메시 구조를 최소 형태로 재현(마스크 좌우
    블록 + 트렌치 오프닝 + 마스크 밑에 숨은 내부 경계선)해 회귀를 막는다."""
    p = np.array(
        [
            [-10.0, 0.0],  # 0: 마스크 밑 숨은 경계(내부) 왼쪽 끝 — 도메인 경계
            [-3.0, 0.0],  # 1: 분기점(왼쪽) — degree 3
            [-10.0, 5.0],  # 2: 마스크 top-left — 도메인 경계
            [-3.0, 5.0],  # 3: 마스크 top-left 모서리
            [3.0, 0.0],  # 4: 분기점(오른쪽) — degree 3
            [3.0, 5.0],  # 5: 마스크 top-right 모서리
            [10.0, 0.0],  # 6: 마스크 밑 숨은 경계(내부) 오른쪽 끝 — 도메인 경계
            [10.0, 5.0],  # 7: 마스크 top-right — 도메인 경계
        ],
        dtype=np.float32,
    )
    lines = np.array(
        [
            [0, 1],  # 내부 경계(왼쪽, 마스크에 덮인 Si 상단) — 제외돼야 함
            [1, 4],  # 트렌치 오프닝 바닥(Si-vacuum, 실제 외곽 경계) — 유지돼야 함
            [4, 6],  # 내부 경계(오른쪽) — 제외돼야 함
            [2, 3],  # 마스크 top(왼쪽)
            [3, 1],  # 마스크 측벽(왼쪽)
            [4, 5],  # 마스크 측벽(오른쪽)
            [5, 7],  # 마스크 top(오른쪽)
        ]
    )
    sdf = signed_dist_field(p, lines, nx=41, ny=41, xr=(-10, 10), yr=(-10, 10))
    gx = np.linspace(-10, 10, 41)
    gy = np.linspace(-10, 10, 41)

    def val(qx: float, qy: float) -> float:
        return float(sdf[np.argmin(np.abs(gy - qy)), np.argmin(np.abs(gx - qx))])

    assert val(-6, 2) < 0  # 마스크 몸통 안쪽(내부 경계 밑에 숨어있던 버그 지점) = solid
    assert val(-6, 10) > 0  # 마스크 위 = vacuum
    assert val(0, 2) > 0  # 트렌치 오프닝 안 = vacuum
    assert val(-6, -2) < 0  # Si 벌크 = solid


def test_signed_dist_field_no_nan() -> None:
    p = np.array([[-10.0, 0.0], [0.0, -3.0], [10.0, 0.0]], dtype=np.float32)
    lines = np.array([[0, 1], [1, 2]])
    sdf = signed_dist_field(p, lines, nx=50, ny=50, xr=(-10, 10), yr=(-10, 10))
    assert not np.isnan(sdf).any()


# --- 픽스처(tests/fixtures/raw) 기반 ----------------------------------------


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_etl_run_on_fixture_produces_graphs(tmp_path: Path) -> None:
    rows = etl_run(FIXTURE_ROOT / "r0001", tmp_path / "r0001", THETA_KEYS)
    assert len(rows) == 4  # K=3 -> t000~t003
    for row in rows:
        assert row["n_nodes"] > 0
    graph_files = sorted((tmp_path / "r0001").glob("g*.npz"))
    assert len(graph_files) == 4


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_depth_is_monotonically_nondecreasing_over_time() -> None:
    """물리 타당성(설계서 §9): 식각 깊이는 시간에 따라 줄어들지 않아야 한다."""
    for rid in FIXTURE_RUNS:
        snaps = sorted((FIXTURE_ROOT / rid / "snapshots").glob("t*.vtp"))
        from etchpipe.etl.etl import load_snapshot

        depths = [metrics(load_snapshot(s)[0])["depth"] for s in snaps]
        assert all(b >= a - 1e-6 for a, b in zip(depths, depths[1:])), (rid, depths)


# --- dataset_hash / manifest ------------------------------------------------


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_dataset_hash_is_deterministic() -> None:
    a = compute_dataset_hash(FIXTURE_RUNS, FIXTURE_ROOT)
    b = compute_dataset_hash(FIXTURE_RUNS, FIXTURE_ROOT)
    assert a == b
    assert len(a) == 12


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_dataset_hash_changes_with_different_run_set() -> None:
    a = compute_dataset_hash(FIXTURE_RUNS, FIXTURE_ROOT)
    b = compute_dataset_hash(FIXTURE_RUNS[:2], FIXTURE_ROOT)
    assert a != b


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_process_dataset_writes_manifest_and_splits(tmp_path: Path) -> None:
    splits = {"train": ["r0000", "r0001"], "val": ["r0002"], "test": [], "ood": []}
    manifest = process_dataset(FIXTURE_ROOT, splits, tmp_path, THETA_KEYS)

    out = tmp_path / manifest["dataset_hash"]
    assert (out / "manifest.json").exists()
    assert (out / "splits.json").exists()
    assert (out / "stats.json").exists()
    assert json.loads((out / "splits.json").read_text()) == splits
    assert manifest["n_recipes"] == 3
    assert manifest["splits_sizes"] == {"train": 2, "val": 1, "test": 0, "ood": 0}
    for rid in ("r0000", "r0001", "r0002"):
        assert (out / "graphs" / rid / "metrics.json").exists()


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_process_dataset_parallel_matches_serial(tmp_path: Path) -> None:
    """W3-09: max_workers>1(ProcessPoolExecutor)과 기본값 1(직렬)의 결과가 바이트 단위로 같아야 한다
    — 레시피 단위 분할만 다를 뿐 각 run_id의 변환 결과는 동일해야 함."""
    splits = {"train": ["r0000", "r0001"], "val": ["r0002"], "test": [], "ood": []}

    out_serial = tmp_path / "serial"
    manifest_serial = process_dataset(FIXTURE_ROOT, splits, out_serial, THETA_KEYS, max_workers=1)

    out_parallel = tmp_path / "parallel"
    manifest_parallel = process_dataset(
        FIXTURE_ROOT, splits, out_parallel, THETA_KEYS, max_workers=2
    )

    assert manifest_serial == manifest_parallel
    dh = manifest_serial["dataset_hash"]
    for rid in ("r0000", "r0001", "r0002"):
        a = (out_serial / dh / "graphs" / rid / "metrics.json").read_bytes()
        b = (out_parallel / dh / "graphs" / rid / "metrics.json").read_bytes()
        assert a == b
        for k in range(4):
            sdf_a = np.load(out_serial / dh / "graphs" / rid / f"sdf{k:03d}.npy")
            sdf_b = np.load(out_parallel / dh / "graphs" / rid / f"sdf{k:03d}.npy")
            assert np.array_equal(sdf_a, sdf_b)


# --- stats.json (W3-04, FR-05: 정규화 통계는 train에서만) --------------------


def _write_fake_graph(
    d: Path, k: int, node: np.ndarray, edge: np.ndarray, sdf: np.ndarray
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"g{k:03d}.npz",
        node=node,
        edge_index=np.zeros((2, 0), np.int64),
        edge=edge,
        pos=node[:, :2],
    )
    np.save(d / f"sdf{k:03d}.npy", sdf)


def test_compute_stats_matches_analytic_mean_std(tmp_path: Path) -> None:
    """해석적으로 답을 아는 입력(설계서 §9): node 값 0과 2 -> 평균 1, 표준편차 1."""
    root = tmp_path / "graphs"
    node = np.array([[0.0] * 14, [2.0] * 14], dtype=np.float32)
    edge = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    sdf = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)  # 평균 0, 표준편차 1
    _write_fake_graph(root / "r0000", 0, node, edge, sdf)

    stats = compute_stats(root, ["r0000"])

    assert np.allclose(stats["node_mean"], [1.0] * 14)
    assert np.allclose(stats["node_std"], [1.0] * 14)
    assert np.allclose(stats["edge_mean"], [2.0, 2.0, 2.0])
    assert np.allclose(stats["edge_std"], [2.0, 2.0, 2.0])
    assert stats["sdf_mean"] == pytest.approx(0.0)
    assert stats["sdf_std"] == pytest.approx(1.0)
    assert stats["n_nodes"] == 2
    assert stats["n_edges"] == 2
    assert stats["n_sdf_values"] == 4


def test_compute_stats_accumulates_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "graphs"
    node_a = np.zeros((1, 14), dtype=np.float32)
    node_b = np.full((1, 14), 4.0, dtype=np.float32)
    edge = np.zeros((1, 3), dtype=np.float32)
    sdf = np.zeros((2, 2), dtype=np.float32)
    _write_fake_graph(root / "r0000", 0, node_a, edge, sdf)
    _write_fake_graph(root / "r0001", 0, node_b, edge, sdf)

    stats = compute_stats(root, ["r0000", "r0001"])
    assert np.allclose(stats["node_mean"], [2.0] * 14)  # (0+4)/2
    assert stats["n_nodes"] == 2


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_process_dataset_stats_uses_train_split_only(tmp_path: Path) -> None:
    splits_a = {"train": ["r0000"], "val": ["r0001"], "test": [], "ood": ["r0002"]}
    out_a = tmp_path / "a"
    manifest_a = process_dataset(FIXTURE_ROOT, splits_a, out_a, THETA_KEYS)
    stats_a = json.loads((out_a / manifest_a["dataset_hash"] / "stats.json").read_text())

    # 같은 run_id 집합이지만 train에 r0001을 더 포함 -> stats가 달라져야 한다
    splits_b = {"train": ["r0000", "r0001"], "val": [], "test": [], "ood": ["r0002"]}
    out_b = tmp_path / "b"
    manifest_b = process_dataset(FIXTURE_ROOT, splits_b, out_b, THETA_KEYS)
    stats_b = json.loads((out_b / manifest_b["dataset_hash"] / "stats.json").read_text())

    assert stats_a["n_nodes"] != stats_b["n_nodes"]
    assert stats_a != stats_b


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_process_dataset_stats_json_is_deterministic(tmp_path: Path) -> None:
    """FR-05: 같은 입력(raw+splits) -> 같은 출력(dataset_hash, stats.json)."""
    splits = {"train": ["r0000", "r0001"], "val": ["r0002"], "test": [], "ood": []}
    m1 = process_dataset(FIXTURE_ROOT, splits, tmp_path / "a", THETA_KEYS)
    m2 = process_dataset(FIXTURE_ROOT, splits, tmp_path / "b", THETA_KEYS)
    assert m1["dataset_hash"] == m2["dataset_hash"]
    s1 = (tmp_path / "a" / m1["dataset_hash"] / "stats.json").read_bytes()
    s2 = (tmp_path / "b" / m2["dataset_hash"] / "stats.json").read_bytes()
    assert s1 == s2


# --- validate_run / validate_dataset (W3-05) --------------------------------


def _write_fake_vtp(
    path: Path, points: np.ndarray, lines: np.ndarray, mat: np.ndarray, normals: np.ndarray
) -> None:
    import pyvista as pv

    conn = np.empty((len(lines), 3), dtype=np.int64)
    conn[:, 0] = 2
    conn[:, 1:] = lines
    pts3 = np.zeros((len(points), 3), dtype=np.float64)
    pts3[:, :2] = points
    mesh = pv.PolyData(pts3, lines=conn.ravel())
    mesh.cell_data["MaterialIds"] = mat
    nrm3 = np.zeros((len(normals), 3), dtype=np.float64)
    nrm3[:, :2] = normals
    mesh.cell_data["Normals"] = nrm3
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_validate_run_passes_on_fixture() -> None:
    r = validate_run(FIXTURE_ROOT / "r0001")
    assert r["errors"] == []
    assert r["n_snapshots"] == 4  # meta.json actual_k=3 -> 4
    assert r["max_n_points"] > 0


@pytest.mark.skipif(not FIXTURE_ROOT.exists(), reason="fixtures 없음")
def test_validate_dataset_aggregates_fixture_runs() -> None:
    report = validate_dataset(FIXTURE_ROOT, FIXTURE_RUNS)
    assert report["n_runs"] == 3
    assert report["n_bad"] == 0
    assert report["max_n_points"] > 0


def test_validate_run_detects_snapshot_count_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "bad_count"
    run.mkdir(parents=True)
    json.dump({"actual_k": 2, "params": {}}, open(run / "meta.json", "w"))
    p = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    lines = np.array([[0, 1]])
    mat = np.array([10])
    nv = np.array([[0.0, 1.0]])
    _write_fake_vtp(run / "snapshots" / "t000.vtp", p, lines, mat, nv)  # actual_k+1=3 기대, 1개뿐

    r = validate_run(run)
    assert any("snapshot_count" in e for e in r["errors"]), r


def test_validate_run_detects_nan_points(tmp_path: Path) -> None:
    run = tmp_path / "bad_nan"
    run.mkdir(parents=True)
    json.dump({"actual_k": 0, "params": {}}, open(run / "meta.json", "w"))
    p = np.array([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float32)
    lines = np.array([[0, 1]])
    mat = np.array([10])
    nv = np.array([[0.0, 1.0]])
    _write_fake_vtp(run / "snapshots" / "t000.vtp", p, lines, mat, nv)

    r = validate_run(run)
    assert any("NaN" in e for e in r["errors"]), r


def test_validate_run_detects_non_monotonic_depth(tmp_path: Path) -> None:
    run = tmp_path / "bad_depth"
    run.mkdir(parents=True)
    json.dump({"actual_k": 1, "params": {}}, open(run / "meta.json", "w"))
    lines = np.array([[0, 1]])
    mat = np.array([10])
    nv = np.array([[0.0, 1.0]])
    p0 = np.array([[0.0, 0.0], [1.0, -5.0]], dtype=np.float32)  # depth=5
    p1 = np.array([[0.0, 0.0], [1.0, -2.0]], dtype=np.float32)  # depth=2 (감소 -> 비정상)
    _write_fake_vtp(run / "snapshots" / "t000.vtp", p0, lines, mat, nv)
    _write_fake_vtp(run / "snapshots" / "t001.vtp", p1, lines, mat, nv)

    r = validate_run(run)
    assert any("monotonic" in e for e in r["errors"]), r
