"""W3-12: ion_mean_energy OOD 음수 버그 수정 후, 영향받은 12건 재생성 전용 1회성 스크립트.
etch-sim:0.1 이미지 컨테이너 안에서 실행한다(/app/src에 run_sim.py가 있음)."""
import json
import pathlib
import sys
import traceback

sys.path.insert(0, "/app/src")
from run_sim import run_sim  # noqa: E402

AFFECTED = [
    "ood0004", "ood0007", "ood0008", "ood0009", "ood0010", "ood0012",
    "ood0013", "ood0014", "ood0015", "ood0017", "ood0018", "ood0019",
]


def main() -> None:
    recipes = json.load(open("/data/recipes.json"))
    by_id = {r["run_id"]: r["params"] for r in recipes}
    raw = pathlib.Path("/data/raw")
    failed = []
    for rid in AFFECTED:
        out = raw / rid
        if (out / "meta.json").exists():
            print("skip", rid, flush=True)
            continue
        try:
            m = run_sim(by_id[rid], out, K=20)
            print("ok", rid, m["wall_sec"], "s", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(rid)
    if failed:
        print("FAILED:", failed, flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
