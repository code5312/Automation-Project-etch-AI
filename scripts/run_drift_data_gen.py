"""W6-03 1단계(`etch-sim:0.1` 컨테이너 전용): 드리프트 데이터 60개(train 40/val 10/
golden_drift 10) 생성 — 루프 1차 시도(ADR-0002)의 입력. `scripts/run_drift_sim.py`와
같은 이유로 ETL·dataset_hash 계산은 별도 단계(`scripts/process_drift_data.py`, train
컨테이너)로 나눈다.

**[확인, 2026-11-12 사용자 결정]**: 이 스크립트는 지금 당장 실행하지 않는다 — W6-04
루프 자체(오케스트레이션 코드)를 먼저 구축하고, 실제 60개 시뮬레이션+미세조정+게이트
실행(증거②)은 나중에 수동으로 트리거한다. 레시피당 8~47분(golden셋 실측, 이번 10개
모니터링 실측은 평균 ~5.6분/레시피로 더 빠름) — 60개면 대략 2~6시간.
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.sim.drift_recipes import drift_splits  # noqa: E402
from etchpipe.sim.run_sim import run_sim  # noqa: E402

DRIFT_RAW_DIR = REPO_ROOT / "data" / "drift" / "raw"
SUBSTRATE_A_IE = 8.05  # configs/sim/live.yaml, 7.0 * 1.15


def main() -> int:
    splits, params_by_id = drift_splits()
    run_ids = [rid for group in splits.values() for rid in group]
    for rid in run_ids:
        out = DRIFT_RAW_DIR / rid
        if (out / "meta.json").exists():
            print(f"skip (이미 있음) {rid}")
            continue
        meta = run_sim(
            params_by_id[rid], out, substrate_a_ie=SUBSTRATE_A_IE, sim_config_id="live_v1"
        )
        print(f"ok {rid}: wall_sec={meta['wall_sec']}, actual_k={meta['actual_k']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
