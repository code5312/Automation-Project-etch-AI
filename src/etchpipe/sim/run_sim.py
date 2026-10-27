"""C1-b: run_sim(params, out_dir) -> 스냅샷 K개 .vtp + meta.json"""
import json, time, sys, hashlib, pathlib
import viennaps as ps

STEP_MIN = 1.0   # 한 스냅샷 간격(분)  [제안]

def run_sim(params: dict, out_dir, K=20, rays=300, seed=42, grid_delta=0.3):
    out = pathlib.Path(out_dir); (out/"snapshots").mkdir(parents=True, exist_ok=True)
    ps.Length.setUnit("um"); ps.Time.setUnit("min")
    domain = ps.Domain(xExtent=30, gridDelta=grid_delta)
    ps.MakeTrench(domain, trenchWidth=params["mask_width"], trenchDepth=0.0,
                  maskHeight=params["mask_height"], maskTaperAngle=10).apply()
    p = ps.SF6O2Etching.defaultParameters()
    p.ionFlux = params["ion_flux"]; p.etchantFlux = params["etchant_flux"]
    p.passivationFlux = params["oxygen_flux"]
    p.Ions.meanEnergy = params["ion_mean_energy"]; p.Ions.sigmaEnergy = 10.0
    p.Ions.exponent = params["ion_exponent"]
    model = ps.SF6O2Etching(p)
    adv = ps.AdvectionParameters(); adv.spatialScheme = ps.SpatialScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER
    rt = ps.RayTracingParameters(); rt.raysPerPoint = rays; rt.useRandomSeeds = False; rt.rngSeed = seed
    cov = ps.CoverageParameters(); cov.tolerance = 1e-4; cov.maxIterations = 10

    t0 = time.time()
    domain.saveSurfaceMesh(str(out/"snapshots"/"t000.vtp"), addInterfaces=True)
    for k in range(1, K+1):
        proc = ps.Process(); proc.setDomain(domain); proc.setProcessModel(model)
        proc.setProcessDuration(STEP_MIN)
        proc.setParameters(adv); proc.setParameters(rt); proc.setParameters(cov)
        proc.apply()
        domain.saveSurfaceMesh(str(out/"snapshots"/f"t{k:03d}.vtp"), addInterfaces=True)
    meta = {"params": params, "K": K, "step_min": STEP_MIN, "rays": rays, "seed": seed,
            "grid_delta": grid_delta, "viennaps": getattr(ps, "__version__", "4.5.0"),
            "wall_sec": round(time.time()-t0, 1),
            "params_hash": hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]}
    json.dump(meta, open(out/"meta.json", "w"), indent=1)
    return meta

if __name__ == "__main__":
    # 사용: python src/run_sim.py <recipes.json> <index> [K]
    rec = json.load(open(sys.argv[1]))[int(sys.argv[2])]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    print(run_sim(rec["params"], f"data/raw/{rec['run_id']}", K=K))
