"""W2-10: gate-negative(더미) — 정상 통과 / 무작위 변위 모델 차단(1) / dataset_hash 불일치 차단(2).

증거 ①(회귀 게이트 차단)의 더미 모델 버전 프로토타입. 5주차(W5-08)에서 실모델로 교체한다.
"""

import json
import pathlib
import sys

import numpy as np

from etchpipe.gate.gate import run_gate
from etchpipe.models.dummy import RandomDisplacementModel, ZeroDisplacementModel

_LINE = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
GOLDEN_PAIRS = [(_LINE, _LINE)]
MAX_CHAMFER = 0.1


def main() -> int:
    cases = []

    r1 = run_gate(ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=MAX_CHAMFER)
    cases.append({"name": "정상 모델(변위 0)", "expected_exit_code": 0, "result": r1})

    r2 = run_gate(RandomDisplacementModel(scale=5.0, seed=0), GOLDEN_PAIRS, max_chamfer=MAX_CHAMFER)
    cases.append({"name": "무작위 변위 모델(망가진 모델)", "expected_exit_code": 1, "result": r2})

    r3 = run_gate(
        ZeroDisplacementModel(), GOLDEN_PAIRS, max_chamfer=MAX_CHAMFER,
        expected_dataset_hash="dataset-v1", actual_dataset_hash="tampered",
    )
    cases.append({"name": "dataset_hash 불일치", "expected_exit_code": 2, "result": r3})

    all_ok = True
    report = []
    for c in cases:
        got = c["result"].exit_code
        ok = got == c["expected_exit_code"]
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        exp = c["expected_exit_code"]
        reasons = c["result"].reasons
        print(f"[{status}] {c['name']}: 기대={exp} 실제={got} reasons={reasons}")
        report.append({
            "name": c["name"], "expected_exit_code": c["expected_exit_code"],
            "actual_exit_code": got, "passed_as_expected": ok, "reasons": c["result"].reasons,
        })

    out = pathlib.Path("build/gate_negative_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)

    if not all_ok:
        print("\ngate-negative 실패: 기대와 다른 case가 있음")
        return 1
    print("\ngate-negative 통과: 3개 케이스 모두 기대대로 동작")
    return 0


if __name__ == "__main__":
    sys.exit(main())
