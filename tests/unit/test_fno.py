"""W6-11: FNO 래퍼 — GPU 팀원이 전달한 `runs/fno_tuned/` 실제 체크포인트로 검증한다.
`runs/`는 `.gitignore` 대상(`mgn_tuned/model.pt`만 예외)이라 `fno_tuned/`는 로컬에만
있다 — 없는 환경(CI 포함)에서는 건너뛴다."""

import pathlib

import numpy as np
import pytest

from etchpipe.models.fno import FNOModel, load_fno_config

RUN_DIR = pathlib.Path(__file__).resolve().parents[2] / "runs" / "fno_tuned"

pytestmark = pytest.mark.skipif(
    not RUN_DIR.exists(), reason="runs/fno_tuned 체크포인트 없음(로컬 전용, .gitignore 대상)"
)


def test_load_fno_config_matches_known_grid() -> None:
    cfg = load_fno_config(RUN_DIR)
    assert cfg["grid"]["ny"] == 128
    assert cfg["grid"]["nx"] == 256
    assert set(cfg["input"]["theta_keys"]) == {
        "ion_flux", "etchant_flux", "oxygen_flux", "ion_mean_energy",
        "ion_exponent", "mask_width", "mask_height",
    }


def test_rollout_and_curve_match_documented_example_depth() -> None:
    # README.md "바로 실행해 보기" 예시와 같은 값(final depth true 33.22 um / pred
    # 33.05 um, 레시피 r0007, 20분 롤아웃) — 래퍼가 teammate의 inference.py를 그대로
    # 호출하는지 확인.
    example_path = RUN_DIR / "example" / "r0007.npz"
    if not example_path.exists():
        pytest.skip("예시 데이터 없음(로컬 전용)")
    ex = np.load(example_path)
    params = dict(zip([str(k) for k in ex["theta_keys"]], ex["theta"], strict=True))

    model = FNOModel(RUN_DIR)
    sdfs = model.rollout_sdf(ex["sdf0"], params, n_steps=len(ex["sdf_true"]) - 1)
    pred_curve = model.curve_from_sdf(sdfs[-1])
    true_curve = model.curve_from_sdf(ex["sdf_true"][-1])

    depth_pred = -float(pred_curve[:, 1].min())
    depth_true = -float(true_curve[:, 1].min())
    assert depth_true == pytest.approx(33.22, abs=0.05)
    assert depth_pred == pytest.approx(33.05, abs=0.05)


def test_predict_curve_from_vtp_returns_point_array() -> None:
    vtp_path = RUN_DIR.parents[1] / "golden" / "raw" / "r0007" / "snapshots" / "t000.vtp"
    if not vtp_path.exists():
        pytest.skip("golden r0007 t000.vtp 없음")
    model = FNOModel(RUN_DIR)
    cfg = model.cfg
    params = {
        k: float(np.mean([r["low"], r["high"]])) for k, r in cfg["input"]["theta_range"].items()
    }
    curve = model.predict_curve(vtp_path, params, n_steps=2)
    assert curve.ndim == 2
    assert curve.shape[1] == 2
    assert len(curve) > 1
