"""W6-02 2단계(`etch-train:0.1` 컨테이너 전용): `scripts/run_drift_sim.py`(sim 컨테이너)가
이미 만들어 둔 `data/drift_monitor/raw/`를 ETL하고, champion 예측과 비교해 자동 반복
루프(ADR-0002) 트리거 여부를 판정한다. 두 스크립트를 나눈 이유는 run_drift_sim.py의
모듈 docstring 참고(viennaps/torch 분리)."""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.etl.etl import THETA_KEYS, etl_run  # noqa: E402
from etchpipe.eval.eval import load_model_a  # noqa: E402
from etchpipe.retrain.monitor import check_drift, mean_chamfer_over_recipes  # noqa: E402

MGN_TUNED_CKPT = REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"
GOLDEN_RAW_DIR = REPO_ROOT / "golden" / "raw"
MONITOR_RAW_DIR = REPO_ROOT / "data" / "drift_monitor" / "raw"
MONITOR_GRAPHS_ROOT = REPO_ROOT / "data" / "drift_monitor" / "graphs"
GOLDEN_GRAPHS_ROOT = REPO_ROOT / "data" / "drift_monitor" / "golden_graphs"


def _golden_recipe_ids() -> list[str]:
    return sorted(p.name for p in GOLDEN_RAW_DIR.iterdir() if p.is_dir())


def _monitor_recipe_ids() -> list[str]:
    return sorted(p.name for p in MONITOR_RAW_DIR.iterdir() if (p / "meta.json").exists())


def main() -> int:
    model, note, norm = load_model_a(MGN_TUNED_CKPT)
    print(f"champion: {note}")

    # ① champion의 골든셋 Chamfer(드리프트 임계값 기준값) — golden/raw는 이미 시뮬레이션된
    # 데이터라 새 ViennaPS 호출 없음.
    for rid in _golden_recipe_ids():
        out = GOLDEN_GRAPHS_ROOT / rid
        if not out.exists():
            etl_run(GOLDEN_RAW_DIR / rid, out, THETA_KEYS)
    golden_result = mean_chamfer_over_recipes(
        model, norm, GOLDEN_GRAPHS_ROOT, GOLDEN_RAW_DIR, _golden_recipe_ids()
    )
    champion_golden_chamfer = golden_result["mean"]
    print(f"champion_golden_chamfer={champion_golden_chamfer:.4f}")

    # ② 모니터링 레시피 ETL(시뮬레이션은 run_drift_sim.py가 이미 끝냄).
    monitor_ids = _monitor_recipe_ids()
    for rid in monitor_ids:
        if not (MONITOR_GRAPHS_ROOT / rid).exists():
            etl_run(MONITOR_RAW_DIR / rid, MONITOR_GRAPHS_ROOT / rid, THETA_KEYS)

    # ③ champion 예측 vs 실제 시뮬레이션 결과 Chamfer 비교, drift_threshold 판정.
    result = check_drift(
        model, norm, champion_golden_chamfer,
        MONITOR_GRAPHS_ROOT, MONITOR_RAW_DIR, monitor_ids,
    )
    print(json.dumps(result, indent=1, ensure_ascii=False))

    out_path = REPO_ROOT / "reports" / "eval" / "W6-02_drift_monitor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
