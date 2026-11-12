"""W6-02 1단계(`etch-sim:0.1` 컨테이너 전용): 신규 레시피 10개를 live(드리프트) 설정으로
시뮬레이션. ETL·모델 비교는 `scripts/run_drift_monitor.py`(train 컨테이너)가 이어서 한다
— `run_sim`(viennaps)과 `rollout`(torch/physicsnemo)을 같은 환경에 같이 설치하지 않는다
(설계서 §6 "환경은 세 개로 분리한다").

레시피당 8~47분(golden셋 실측) — 10개면 수시간 걸림 [확인, 2026-11-12 사용자 결정].
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.sim.run_sim import run_sim  # noqa: E402
from etchpipe.sim.sampling import make_recipes  # noqa: E402

MONITOR_RAW_DIR = REPO_ROOT / "data" / "drift_monitor" / "raw"

N_MONITOR = 10
SEED = 9001  # make_recipes 기본 학습 시드(20261027/+1)와 겹치지 않게 임의로 분리
PREFIX = "mon"
SUBSTRATE_A_IE = 8.05  # configs/sim/live.yaml, 7.0 * 1.15


def main() -> int:
    recipes = make_recipes(N_MONITOR, seed=SEED, prefix=PREFIX)
    for r in recipes:
        rid = r["run_id"]
        out = MONITOR_RAW_DIR / rid
        if (out / "meta.json").exists():
            print(f"skip (이미 있음) {rid}")
            continue
        meta = run_sim(r["params"], out, substrate_a_ie=SUBSTRATE_A_IE, sim_config_id="live_v1")
        print(f"ok {rid}: wall_sec={meta['wall_sec']}, actual_k={meta['actual_k']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
