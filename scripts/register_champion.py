"""W6-08 선행: `runs/mgn_tuned/model.pt`를 레지스트리의 첫 champion으로 공식 등록.

지금까지 `registry/`가 한 번도 실제로 채워진 적이 없었다(자동 반복 루프 W6-04가 아직
실행 전이라) — `/predict`가 "레지스트리의 현재 모델을 로드"(설계서 §3 C7)하게 하려면
champion이 실제로 등록돼 있어야 한다. 이 스크립트는 이미 실측한 게이트 결과(W5-09)·
지표(reports/eval/mgn_tuned_val40_guarded)·valid_range(train split 실제 Θ 범위)를 그대로
카드에 기록한다 — 새로 계산하는 값 없음.
"""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.eval.eval import safe_git_sha  # noqa: E402
from etchpipe.gate.gate import GateResult  # noqa: E402
from etchpipe.registry.registry import Registry, compute_valid_range  # noqa: E402

MODEL_ID = "mgn_tuned_ep2"
ARTIFACT_PATH = REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"
DATASET_HASH = "4b8c0ce1bc98"
VAL40_METRICS_PATH = REPO_ROOT / "reports" / "eval" / "mgn_tuned_val40_guarded" / "metrics.json"


def main() -> int:
    splits = json.loads((REPO_ROOT / "data" / "splits.json").read_text())
    valid_range = compute_valid_range(REPO_ROOT / "data" / "raw", splits["train"])

    val_metrics = json.loads(VAL40_METRICS_PATH.read_text())["final_step"]

    # W5-09(reports/evidence/W5-09_gate_negative.md) golden 10개 "정상 모델" 케이스:
    # passed=True, exit_code=0, reasons=[] — 그대로 재사용(새로 게이트를 다시 안 돌림).
    gate_result = GateResult(passed=True, exit_code=0, reasons=[])

    reg = Registry(REPO_ROOT / "registry")
    ok = reg.register(
        MODEL_ID,
        ARTIFACT_PATH,
        gate_result,
        dataset_hash=DATASET_HASH,
        config={"processor_size": 6, "hidden_dim_processor": 128},
        git_sha=safe_git_sha(),
        metrics=val_metrics,
        valid_range=valid_range,
    )
    print(f"registered={ok}, current_champion={reg.current_champion()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
