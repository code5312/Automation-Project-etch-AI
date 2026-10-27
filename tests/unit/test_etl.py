"""W2-04: ETL(그래프 변환, dataset_hash, manifest.json) 단위 테스트."""

import json
from pathlib import Path

import numpy as np
import pytest

from etchpipe.etl.etl import (
    THETA_KEYS,
    compute_dataset_hash,
    dist_field,
    etl_run,
    metrics,
    process_dataset,
    to_graph,
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
    assert json.loads((out / "splits.json").read_text()) == splits
    assert manifest["n_recipes"] == 3
    assert manifest["splits_sizes"] == {"train": 2, "val": 1, "test": 0, "ood": 0}
    for rid in ("r0000", "r0001", "r0002"):
        assert (out / "graphs" / rid / "metrics.json").exists()
