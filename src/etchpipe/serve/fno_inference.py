"""Serve a registered FNO using the shared recipe response schema."""

import math
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from etchpipe.serve.inference import MAX_STEPS, STEP_MIN, is_in_range, write_initial_surface

PROCESS_KEY = "substrate_a_ie"


def load_fno(run_dir: Path, card: dict):
    from etchpipe.models.fno import FNOModel

    import_module("skimage.measure")
    model = FNOModel(run_dir, device="cpu")
    cfg = model.cfg
    if cfg["step_minutes"] != STEP_MIN or PROCESS_KEY not in cfg["input"]["theta_keys"]:
        raise ValueError("Unsupported registered FNO input schema")
    nominal = card["config"]["nominal_substrate_a_ie"]
    if not is_in_range({PROCESS_KEY: nominal}, card["valid_range"]):
        raise ValueError("FNO nominal A_ie lies outside its registered range")
    return model


def predict_fno(champion: dict, theta_dict: dict[str, float], times: list[float]) -> dict:
    from etchpipe.eval.metrics import cd_profile, depth, mask_from_points, taper_angle_deg

    model, card = champion["model"], champion["card"]
    params = dict(theta_dict)
    params.setdefault(PROCESS_KEY, card["config"]["nominal_substrate_a_ie"])
    with TemporaryDirectory() as scratch_dir:
        initial = Path(scratch_dir) / "t000.vtp"
        write_initial_surface(initial, params["mask_width"], params["mask_height"])
        sdf0 = model.sdf_from_vtp(initial)
    raw_steps = [max(1, round(value / STEP_MIN)) for value in times]
    steps = [min(step, MAX_STEPS) for step in raw_steps]
    trajectory = model.rollout_sdf(sdf0, params, n_steps=max(steps))
    if not np.isfinite(trajectory).all():
        raise ValueError("FNO returned a nonfinite trajectory")
    curves = [model.curve_from_sdf(trajectory[step]) for step in steps]
    if any(len(curve) < 2 or not np.isfinite(curve).all() for curve in curves):
        raise ValueError("FNO returned invalid profiles")
    profiles = [{"t": round(step * STEP_MIN, 6), "points": curve.tolist()}
                for step, curve in zip(steps, curves, strict=True)]
    curve = curves[-1]
    grid = model.cfg["grid"]
    xr, yr = (grid["x_min_um"], grid["x_max_um"]), (grid["y_bottom_um"], grid["y_top_um"])
    vertices = np.arange(len(curve))
    edges = np.column_stack((vertices[:-1], vertices[1:]))
    depth_um = depth(curve)
    widths = cd_profile(mask_from_points(curve, edges, xr=xr, yr=yr), xr, yr, depth_um)
    taper = taper_angle_deg(widths["top"], widths["bot"], depth_um)
    if not all(math.isfinite(value) for value in (depth_um, widths["mid"], taper)):
        raise ValueError("FNO returned invalid profile metrics")
    return {
        "model_version": champion["model_id"],
        "profiles": profiles,
        "cd": widths["mid"],
        "depth": depth_um,
        "taper_deg": taper,
        "in_range": is_in_range(params, card["valid_range"]),
        "clamped": any(step > MAX_STEPS for step in raw_steps),
    }
