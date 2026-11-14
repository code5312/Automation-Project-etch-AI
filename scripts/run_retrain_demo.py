"""W6-07: `retrain-demo` — C8 재학습 시나리오 전체를 소규모 설정으로 시연.

`.github/workflows/retrain-demo.yml`의 `workflow_dispatch` 수동 트리거로 실행된다
(설계서 §8: "수동 호출(드리프트 데이터 지정) → C8 재학습 시나리오 전체"). 실제 W6-03~06
의 60개 대신 아주 작은 드리프트셋(기본 train 2/val 1/golden_drift 1)으로 **같은
ADR-0002 루프·같은 콜백**(`retrain/execute.py`)을 실제로 돈다 — 실제 champion
(`runs/mgn_tuned/model.pt`)·실제 ViennaPS 시뮬레이션·실제 게이트를 쓰되, 결과를
production `registry/`에는 **쓰지 않는다**(데모 산출물은 전부 `build/retrain_demo/`
전용 경로에만 남긴다 — 실제 증거는 W6-06 참고).

사용: PYTHONPATH=src python scripts/run_retrain_demo.py
환경변수로 크기 조정 가능(모두 선택): N_TRAIN, N_VAL, N_GOLDEN_DRIFT (기본 2/1/1)
출력: build/retrain_demo/result.json
"""

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from etchpipe.eval.eval import load_model_a
from etchpipe.retrain.execute import (
    CHAMPION_CKPT,
    SUBSTRATE_A_IE,
    DriftState,
    make_fine_tune_fn,
    make_gate_fn,
)
from etchpipe.retrain.loop import ATTEMPTS, run_retrain_loop
from etchpipe.sim.drift_recipes import drift_splits
from etchpipe.sim.run_sim import run_sim

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "build" / "retrain_demo"
DEMO_SEED, DEMO_PREFIX = 9500, "demo"


def _make_scaled_regenerate_fn(state: DriftState, initial_n_train: int):
    """`loop.py::ATTEMPTS`의 2차 시도 `n_train`(100, 실제 운영 40→100 기준)은 데모
    규모에 안 맞다 — 그대로 넘기면 데모 train 2개를 100개로 확장하려 들어 ViennaPS를
    98번 추가로 돌리는 폭주가 생긴다(2026-11-13 실측으로 발견, 실행 중 중단함). 실제
    운영의 확장 비율(40→100, ×2.5)을 그대로 데모 크기에 적용해 작게 유지한다[확인,
    실측 버그 수정]."""
    base_n_train = ATTEMPTS[0]["n_train"]

    def regenerate_fn(n_target_real: int) -> None:
        ratio = n_target_real / base_n_train
        n_target_demo = max(initial_n_train + 1, round(initial_n_train * ratio))
        n_added = state.expand_train(n_target_demo)
        print(
            f"  regenerate_fn(데모 축소): 실제 목표 {n_target_real} -> 데모 목표 "
            f"{n_target_demo}, train +{n_added}(now {len(state.splits['train'])})"
        )

    return regenerate_fn


def _timed(fn, bucket: list[float]):
    """W7-07: 호출마다 걸린 시간을 `bucket`에 누적만 하는 투명 래퍼(fine_tune_fn/gate_fn을
    `run_retrain_loop`에 주입하기 전에 감싼다 — loop.py/execute.py 자체는 건드리지 않음)."""

    def wrapped(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        bucket.append(time.time() - t0)
        return result

    return wrapped


def main() -> int:
    n_train = int(os.environ.get("N_TRAIN", "2"))
    n_val = int(os.environ.get("N_VAL", "1"))
    n_golden_drift = int(os.environ.get("N_GOLDEN_DRIFT", "1"))
    raw_dir = DEMO_ROOT / "raw"
    pipeline_t0 = time.time()

    print(f"데모 드리프트 레시피 {n_train + n_val + n_golden_drift}개 ViennaPS 시뮬레이션 중...")
    # `DriftState`는 원래(W6-03/04) raw_dir에 meta.json이 이미 있다고 가정한다
    # (실 운영에선 run_drift_data_gen.py가 별도로 먼저 돌려둠) — 데모는 이 스크립트가
    # 직접 처음 생성까지 포함한다. 같은 (n/seed/prefix)로 drift_splits()를 두 번(여기·
    # DriftState 내부) 호출해도 결정적이라 같은 레시피가 나온다[확인].
    sim_t0 = time.time()
    splits, params_by_id = drift_splits(n_train, n_val, n_golden_drift, DEMO_SEED, DEMO_PREFIX)
    n_simulated = 0
    for group in splits.values():
        for rid in group:
            out = raw_dir / rid
            if not (out / "meta.json").exists():
                meta = run_sim(
                    params_by_id[rid], out, substrate_a_ie=SUBSTRATE_A_IE, sim_config_id="live_v1"
                )
                n_simulated += 1
                print(f"  ok {rid}: wall_sec={meta['wall_sec']}, actual_k={meta['actual_k']}")
    sim_sec = time.time() - sim_t0

    print("ETL 중...")
    etl_t0 = time.time()
    state = DriftState(
        raw_dir=raw_dir,
        processed_root=DEMO_ROOT / "processed",
        n_train=n_train, n_val=n_val, n_golden_drift=n_golden_drift,
        seed=DEMO_SEED, prefix=DEMO_PREFIX,
        expand_seed=9600, expand_prefix="demox",
    )
    etl_sec = time.time() - etl_t0
    print(f"  train={len(state.splits['train'])} val={len(state.splits['val'])} "
          f"golden_drift={len(state.splits['golden_drift'])} dataset_hash={state.dataset_hash}")

    _model, _note, champion_norm = load_model_a(CHAMPION_CKPT)

    fine_tune_sec_by_attempt: list[float] = []
    gate_sec_by_attempt: list[float] = []
    result = run_retrain_loop(
        CHAMPION_CKPT,
        _timed(
            make_fine_tune_fn(state, champion_norm, out_root=DEMO_ROOT / "runs"),
            fine_tune_sec_by_attempt,
        ),
        _timed(make_gate_fn(state), gate_sec_by_attempt),
        regenerate_fn=_make_scaled_regenerate_fn(state, n_train),
    )
    total_sec = time.time() - pipeline_t0

    timings = {
        "sim_sec": round(sim_sec, 1),
        "sim_recipes_newly_simulated": n_simulated,
        "etl_sec": round(etl_sec, 1),
        "fine_tune_sec_by_attempt": [round(s, 1) for s in fine_tune_sec_by_attempt],
        "gate_sec_by_attempt": [round(s, 1) for s in gate_sec_by_attempt],
        "total_sec": round(total_sec, 1),
    }
    print(f"\n단계별 실행 시간(W7-07): {timings}")

    summary = {
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "note": "[제안] 소규모 데모 — registry에 승격하지 않음, 실제 증거는 W6-06 참고",
        "n_recipes": {"train": n_train, "val": n_val, "golden_drift": n_golden_drift},
        "dataset_hash": state.dataset_hash,
        "promoted": result["promoted"],
        "timings": timings,
        "attempts": [
            {"attempt": a["attempt"], "config": a["config"], "gate_result": a["gate_result"]}
            for a in result["attempts"]
        ],
    }
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = DEMO_ROOT / "result.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out_path}")
    print(f"promoted={result['promoted']} (데모 — registry 승격 안 함)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
