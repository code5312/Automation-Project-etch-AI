# W3-01-GPU 스파이크: Colab GPU에서 ViennaPS 레이트레이싱 가속 검증용 스크립트.
#
# 사용법: 이 파일 내용을 Colab 노트북에 "# %%" 단위로 셀마다 복사해 순서대로 실행한다.
# 목적: r0004(가장 높은 ion_flux, CPU 기준 확보됨)를 GPU 빌드로 동일 조건 재실행해
#       (a) 물리 결과가 CPU와 얼마나 다른지, (b) 속도 이득이 실제로 있는지 확인한다.
# 시간상자: 반나절. 넘기면 중단하고 reports/evidence/W3-01-GPU_spike.md에 중단 사유를 적는다.
# 판정 기준·결과 기록: reports/evidence/W3-01-GPU_spike.md
#
# CPU 기준값 (reports/evidence/W3-01_depth_max_revalidation.md, 2026-10-28 실측):
#   actual_k=5, final_depth_um=55.5, wall_sec=386.2, params_hash=371f4e374a, stopped_early=True

# %% [셀 1] GPU/드라이버/컴파일러 확인 — 요구사항(ViennaPS 공식 문서 기준): 드라이버 570+, CUDA 12.0+, GCC 12+
# 이 셀 결과가 요구사항 미달이면 여기서 중단하고 판정표에 "환경 불충분"으로 기록한다.
"""
!nvidia-smi
!nvcc --version
!gcc --version
"""

# %% [셀 2] 선행 조건 — 사용자가 직접 해야 함 (스크립트로 대체 불가)
# 1. https://developer.nvidia.com/designworks/optix/download 에서 OptiX SDK 8.0 다운로드
#    (NVIDIA Developer 계정 가입/로그인 필요 — 본인이 직접 진행. CLAUDE.md "외부 서비스 가입" 항목)
# 2. Colab 좌측 파일 탭에 다운받은 .sh 또는 압축파일 업로드 → 압축 해제
# 3. 아래 OPTIX_DIR을 실제 압축 해제 경로로 수정
"""
import os
OPTIX_DIR = "/content/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"  # 실제 경로로 수정
assert os.path.isdir(OPTIX_DIR), f"OptiX SDK 경로 확인 필요: {OPTIX_DIR}"
os.environ["OptiX_INSTALL_DIR"] = OPTIX_DIR
"""

# %% [셀 3] 빌드 의존성 (CLAUDE.md 금지 버전: ViennaLS 5.9.0 사용 금지 — 5.8.5로 고정)
"""
!apt-get -qq update
!apt-get -qq install -y libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6 cmake ninja-build
!pip install --no-cache-dir numpy==2.5.4 scipy==1.18.1
"""

# %% [셀 4] ViennaLS 5.8.5 소스 빌드 (CPU 베이스 — GPU 가속은 ViennaPS 쪽 레이트레이싱)
"""
!git clone --branch v5.8.5 --depth 1 https://github.com/ViennaTools/ViennaLS.git /content/ViennaLS
%cd /content/ViennaLS
!pip install .
%cd /content
"""

# %% [셀 5] ViennaPS 4.5.0 소스 빌드, GPU 플래그 ON
# 주의: 공식 문서 기준 CMAKE_ARGS=-DVIENNAPS_USE_GPU=ON pip install . 경로.
# 빌드 실패 시(흔함) 여기서 중단하고 판정표에 에러 로그 요약을 남긴다 — 무리하게 디버깅에
# 시간상자를 넘기지 않는다.
"""
import os
os.environ["CMAKE_ARGS"] = "-DVIENNAPS_USE_GPU=ON"
!git clone --branch v4.5.0 --depth 1 https://github.com/ViennaTools/ViennaPS.git /content/ViennaPS
%cd /content/ViennaPS
!pip install --no-deps .
# --no-deps: pyproject.toml의 dependencies=["ViennaLS>=5.7.2"] 때문에 pip가 의존성을
# 다시 확인하면서 셀 4에서 소스로 빌드해 둔 ViennaLS를 PyPI 휠로 덮어쓸 위험을 막음
# (공식 설치 스크립트 python/scripts/install_ViennaPS.py도 이 옵션을 씀, 2026-11-04 확인).
%cd /content
"""

# %% [셀 6] API 확인 — GPU 레이트레이싱을 실제로 어떻게 켜는지 [미확인], 추측하지 않고 직접 조사
"""
import viennaps as ps
print("RayTracingParameters 속성:", [x for x in dir(ps.RayTracingParameters) if not x.startswith("_")])
print("ps 모듈에서 GPU/Ray 관련 이름:", [x for x in dir(ps) if "GPU" in x.upper() or "Ray" in x])
print("viennaps 버전:", getattr(ps, "__version__", "?"))
"""

# %% [셀 7] r0004를 CPU(run_sim.py)와 동일 코드·동일 파라미터로 실행
# src/etchpipe/sim/run_sim.py와 완전히 동일한 로직 — 물리 결과를 비교하려면 로직이 달라지면 안 됨.
"""
import hashlib, json, time, pathlib
import viennaps as ps

STEP_MIN = 1.0
DEPTH_MAX_DEFAULT = 50.0

def _current_depth(domain):
    (_, y_min, _), _ = domain.getBoundingBox()
    return -y_min

def run_sim(params, out_dir, K=20, rays=300, seed=42, grid_delta=0.3, depth_max=DEPTH_MAX_DEFAULT):
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
        "params": params, "K": K, "actual_k": actual_k, "step_min": STEP_MIN,
        "rays": rays, "seed": seed, "grid_delta": grid_delta, "depth_max": depth_max,
        "stopped_early": stopped_early,
        "final_depth_um": round(_current_depth(domain), 3),
        "viennaps": getattr(ps, "__version__", "4.5.0"),
        "wall_sec": round(time.time() - t0, 1),
        "params_hash": hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10],
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    return meta


R0004_PARAMS = {
    "ion_flux": 25.90269959878192,
    "etchant_flux": 9081.971904011029,
    "oxygen_flux": 416.88178219004146,
    "ion_mean_energy": 132.22102271392941,
    "ion_exponent": 672.3899119017115,
    "mask_width": 13.302675507962704,
    "mask_height": 10.660476841032505,
}

CPU_BASELINE = {
    "actual_k": 5, "final_depth_um": 55.5, "wall_sec": 386.2,
    "params_hash": "371f4e374a", "stopped_early": True,
}

result = run_sim(R0004_PARAMS, "/content/out/r0004_gpu")
print("GPU 결과:", json.dumps(result, indent=1))
print("CPU 기준(W3-01_depth_max_revalidation.md):", CPU_BASELINE)
assert result["params_hash"] == CPU_BASELINE["params_hash"], "params_hash 불일치 — 파라미터가 다름, 비교 무효"
print("actual_k 일치:", result["actual_k"] == CPU_BASELINE["actual_k"])
print("final_depth_um 차이:", round(abs(result["final_depth_um"] - CPU_BASELINE["final_depth_um"]), 3))
print("속도 배율(CPU wall_sec / GPU wall_sec):", round(CPU_BASELINE["wall_sec"] / result["wall_sec"], 2))
"""

# %% [셀 8] (선택) 3D 포인트 단위 비교 — final_depth_um만으로 부족하면 Chamfer까지 확인
# tests/unit/test_metrics.py의 chamfer_distance를 같은 방식으로 Colab에 복사해
# t000~t_actual_k 스냅샷 전체를 CPU 산출물(로컬에 남아있다면)과 비교. 1차 판정에서
# final_depth_um 차이가 이미 크면(예: >1um) 이 단계는 생략해도 된다.
