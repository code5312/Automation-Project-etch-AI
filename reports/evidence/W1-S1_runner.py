import json
import pathlib
import sys
import time

sys.path.insert(0, "src/etchpipe/sim")
from sampling import make_recipes
from run_sim import run_sim

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
RAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 300

OUT = pathlib.Path("reports/evidence/W1-S1_timing.json")
recipes = make_recipes(N, seed=20261027)
results = json.loads(OUT.read_text()) if OUT.exists() else []
done_ids = {r["run_id"] for r in results}

t0 = time.time()
for r in recipes:
    if r["run_id"] in done_ids:
        continue
    rt0 = time.time()
    meta = run_sim(r["params"], f"data/raw/{r['run_id']}", K=K, rays=RAYS)
    dt = time.time() - rt0
    snaps = sorted(pathlib.Path(f"data/raw/{r['run_id']}/snapshots").glob("t*.vtp"))
    last = snaps[-1].read_text(encoding="utf-8", errors="ignore") if snaps else ""
    # 깊이는 vtp를 pyvista 없이 바로 파싱하지 않고 run_sim의 meta만 기록 -> 깊이는 별도 스크립트에서 pyvista로 집계
    row = {
        "run_id": r["run_id"],
        "params": r["params"],
        "wall_sec": round(dt, 1),
        "meta_wall_sec": meta["wall_sec"],
        "K": K,
        "rays": RAYS,
    }
    print(row, flush=True)
    results.append(row)
    OUT.write_text(json.dumps(results, indent=1))  # 매 레시피마다 즉시 저장(중단돼도 보존)

print("TOTAL_WALL_SEC", round(time.time() - t0, 1), "DONE", len(results), "/", N)
