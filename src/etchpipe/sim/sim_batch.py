"""C1 배치 실행기. 쿠버네티스 Indexed Job 의 파드 1개 = 샤드 1개를 맡는다.
   샤드 번호: 환경변수 JOB_COMPLETION_INDEX (쿠버네티스가 자동 주입), 없으면 인자로.
   레시피 i 는 (i % SHARDS == 샤드번호) 일 때 이 파드가 처리. 이미 끝난 건(meta.json 있음)은 건너뜀."""
import json, os, sys, time, pathlib, traceback
from run_sim import run_sim

def main():
    shard = int(os.environ.get("JOB_COMPLETION_INDEX", sys.argv[1] if len(sys.argv) > 1 else 0))
    shards = int(os.environ.get("SIM_SHARDS", 1))
    K = int(os.environ.get("SIM_K", 20))
    recipes = json.load(open(os.environ.get("SIM_RECIPES", "data/recipes.json")))
    raw = pathlib.Path(os.environ.get("SIM_RAW", "data/raw"))
    mine = [r for i, r in enumerate(recipes) if i % shards == shard]
    print(f"shard {shard}/{shards}: {len(mine)} recipes, K={K}", flush=True)
    failed = []
    for r in mine:
        out = raw / r["run_id"]
        if (out / "meta.json").exists():
            print("skip", r["run_id"], flush=True); continue
        for attempt in (1, 2):                      # 실패 시 1회 재시도
            try:
                m = run_sim(r["params"], out, K=K); print("ok", r["run_id"], m["wall_sec"], "s", flush=True); break
            except Exception:
                traceback.print_exc()
                if attempt == 2: failed.append(r["run_id"])
    if failed:
        (raw / f"failed_shard{shard}.json").write_text(json.dumps(failed)); sys.exit(1)

if __name__ == "__main__":
    main()
