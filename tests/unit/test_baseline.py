"""W3-07: 베이스라인 2종 단위 테스트. ZeroDisplacementModel은 W2-05 기존 것 재사용(여기서는
DepthMlpExtrapolationModel만 신규 검증), tests/fixtures/raw의 작은 실제 데이터(K=3, 3건)로 확인.
"""

import pathlib

import numpy as np
import pytest

from etchpipe.etl.etl import THETA_KEYS, load_snapshot
from etchpipe.models.baseline import DepthMlpExtrapolationModel
from etchpipe.models.dummy import ZeroDisplacementModel

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _load_run(run_id: str) -> tuple[dict, np.ndarray, np.ndarray]:
    import json

    run = FIXTURES / run_id
    meta = json.loads((run / "meta.json").read_text())
    pos0, *_ = load_snapshot(run / "snapshots" / "t000.vtp")
    k = meta["actual_k"]
    pos_final, *_ = load_snapshot(run / "snapshots" / f"t{k:03d}.vtp")
    return meta, pos0, pos_final


@pytest.fixture
def fixture_runs() -> list[str]:
    return ["r0000", "r0001", "r0002"]


def test_zero_displacement_returns_input_unchanged(fixture_runs) -> None:
    model = ZeroDisplacementModel()
    for rid in fixture_runs:
        _meta, pos0, _true = _load_run(rid)
        pred = model.predict(pos0)
        assert np.array_equal(pred, pos0)


def test_depth_mlp_fits_training_depths_reasonably(fixture_runs) -> None:
    """train 레시피 3개로 과적합: 학습에 쓴 레시피 자체의 깊이 예측 오차가 작아야 한다
    (CLAUDE.md 규칙 3 취지 — 먼저 작은 집합에서 모델이 말이 되는지 확인)."""
    metas = [_load_run(rid)[0] for rid in fixture_runs]
    thetas = [m["params"] for m in metas]
    depths = [m["final_depth_um"] for m in metas]

    model = DepthMlpExtrapolationModel(THETA_KEYS, hidden=8, seed=0)
    model.fit(thetas, depths, epochs=800, lr=0.05)

    preds = [model.predict_depth(t) for t in thetas]
    for pred, true in zip(preds, depths):
        assert abs(pred - true) < 1.0, (pred, true)


def test_depth_mlp_predict_shifts_only_etched_points(fixture_runs) -> None:
    metas = [_load_run(rid)[0] for rid in fixture_runs]
    thetas = [m["params"] for m in metas]
    depths = [m["final_depth_um"] for m in metas]
    model = DepthMlpExtrapolationModel(THETA_KEYS, hidden=8, seed=0)
    model.fit(thetas, depths, epochs=200)

    _meta, pos0, _true = _load_run("r0000")
    pred = model.predict(pos0, thetas[0])
    above = pos0[:, 1] >= 0
    assert np.array_equal(pred[above], pos0[above])
    below = ~above
    assert np.allclose(pred[below, 0], pos0[below, 0])
    assert (pred[below, 1] <= pos0[below, 1]).all()


def test_depth_mlp_predict_depth_never_negative(fixture_runs) -> None:
    metas = [_load_run(rid)[0] for rid in fixture_runs]
    thetas = [m["params"] for m in metas]
    model = DepthMlpExtrapolationModel(THETA_KEYS, hidden=4, seed=1)
    model.fit(thetas, [-1000.0, -1000.0, -1000.0], epochs=50)
    _meta, pos0, _true = _load_run("r0000")
    pred = model.predict(pos0, thetas[0])
    assert (pred[:, 1] <= pos0[:, 1]).all()
    assert np.allclose(pred[pos0[:, 1] >= 0], pos0[pos0[:, 1] >= 0])


def test_depth_mlp_predict_raises_before_fit() -> None:
    model = DepthMlpExtrapolationModel(THETA_KEYS)
    with pytest.raises(RuntimeError):
        model.predict_depth({k: 1.0 for k in THETA_KEYS})
