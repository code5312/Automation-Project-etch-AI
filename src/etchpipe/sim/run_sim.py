"""C1: run_sim(params, out_dir) -> 스냅샷 K개(또는 depth_max 조기 종료) .vtp + meta.json.

W1-S1에서 고정 K로는 일부 레시피가 깊이가 폭주함을 확인(설계서 §3 C1, 정의서 §7.1).
depth_max 정지조건(기본 50µm, W1 게이트 G1·사용자 결정 2026-10-28)으로 막는다.
"""

import hashlib
import json
import pathlib
import sys
import time

import viennaps as ps

STEP_MIN = 1.0  # 한 스냅샷 간격(분) [제안]
DEPTH_MAX_DEFAULT = 50.0  # µm [확인 필요: 실측 기반 제안, W1-S1 참조]


def _current_depth(domain: "ps.Domain") -> float:
    """Si 윗면 y=0 기준 식각 깊이(µm). bounding box의 y 최솟값으로 계산(pyvista 불필요)."""
    (_, y_min, _), _ = domain.getBoundingBox()
    return -y_min


def run_sim(
    params: dict,
    out_dir: str | pathlib.Path,
    K: int = 20,
    rays: int = 300,
    seed: int = 42,
    grid_delta: float = 0.3,
    depth_max: float = DEPTH_MAX_DEFAULT,
) -> dict:
    out = pathlib.Path(out_dir)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)

    ps.Length.setUnit("um")
    ps.Time.setUnit("min")
    domain = ps.Domain(xExtent=30, gridDelta=grid_delta)
    ps.MakeTrench(
        domain, trenchWidth=params["mask_width"], trenchDepth=0.0,
        maskHeight=params["mask_height"], maskTaperAngle=10,
    ).apply()

    p = ps.SF6O2Etching.defaultParameters()
    p.ionFlux = params["ion_flux"]
    p.etchantFlux = params["etchant_flux"]
    p.passivationFlux = params["oxygen_flux"]
    p.Ions.meanEnergy = params["ion_mean_energy"]
    p.Ions.sigmaEnergy = 10.0
    p.Ions.exponent = params["ion_exponent"]
    model = ps.SF6O2Etching(p)

    adv = ps.AdvectionParameters()
    adv.spatialScheme = ps.SpatialScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER
    rt = ps.RayTracingParameters()
    rt.raysPerPoint = rays
    rt.useRandomSeeds = False
    rt.rngSeed = seed
    cov = ps.CoverageParameters()
    cov.tolerance = 1e-4
    cov.maxIterations = 10

    t0 = time.time()
    domain.saveSurfaceMesh(str(out / "snapshots" / "t000.vtp"), addInterfaces=True)

    stopped_early = False
    actual_k = K
    for k in range(1, K + 1):
        proc = ps.Process()
        proc.setDomain(domain)
        proc.setProcessModel(model)
        proc.setProcessDuration(STEP_MIN)
        proc.setParameters(adv)
        proc.setParameters(rt)
        proc.setParameters(cov)
        proc.apply()
        domain.saveSurfaceMesh(str(out / "snapshots" / f"t{k:03d}.vtp"), addInterfaces=True)

        if _current_depth(domain) >= depth_max:
            stopped_early = True
            actual_k = k
            break

    meta = {
        "params": params,
        "K": K,
        "actual_k": actual_k,
        "step_min": STEP_MIN,
        "rays": rays,
        "seed": seed,
        "grid_delta": grid_delta,
        "depth_max": depth_max,
        "stopped_early": stopped_early,
        "final_depth_um": round(_current_depth(domain), 3),
        "viennaps": getattr(ps, "__version__", "4.5.0"),
        "wall_sec": round(time.time() - t0, 1),
        "params_hash": hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10],
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    # 사용: python src/etchpipe/sim/run_sim.py <recipes.json> <index> [K]
    rec = json.load(open(sys.argv[1]))[int(sys.argv[2])]
    k_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    print(run_sim(rec["params"], f"data/raw/{rec['run_id']}", K=k_arg))
