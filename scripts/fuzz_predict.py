"""W6-09: `/predict` 이상 입력 100개 퍼징 — 5xx가 하나도 없는지 확인(실제 champion
필요, `etch-serve:0.1` 컨테이너에서 실행). 음수/0/상한초과/NaN/Infinity를 7개 필드와
times에 각각 주입하고, 마스크 치수 경계값·극단 배치도 섞는다.

사용: PYTHONPATH=src python scripts/fuzz_predict.py
"""

import json
import math
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from etchpipe.serve.app import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)
API_KEY = "test-key"

BASE = {
    "ion_flux": 10.0, "etchant_flux": 4500.0, "oxygen_flux": 800.0,
    "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0,
}
FIELDS = list(BASE.keys())


def cases() -> list[tuple[str, dict, list[float]]]:
    out: list[tuple[str, dict, list[float]]] = []
    for f in FIELDS:
        out.append((f"{f}_negative", dict(BASE, **{f: -1.0}), [1.0]))
        out.append((f"{f}_zero", dict(BASE, **{f: 0.0}), [1.0]))
        out.append((f"{f}_huge", dict(BASE, **{f: 1e12}), [1.0]))
        out.append((f"{f}_nan", dict(BASE, **{f: math.nan}), [1.0]))
        out.append((f"{f}_inf", dict(BASE, **{f: math.inf}), [1.0]))
    out.append(("mask_both_near_ceiling", dict(BASE, mask_width=24.9, mask_height=24.9), [1.0]))
    out.append(("mask_both_at_ceiling", dict(BASE, mask_width=25.0, mask_height=25.0), [1.0]))
    out.append(("times_huge", dict(BASE), [1e9]))
    out.append(("times_negative", dict(BASE), [-5.0]))
    out.append(("times_zero", dict(BASE), [0.0]))
    out.append(("times_nan", dict(BASE), [math.nan]))
    out.append(("times_many_small", dict(BASE), [0.1] * 20))
    out.append(("all_fields_negative", {k: -v for k, v in BASE.items()}, [1.0]))
    out.append(("all_fields_huge", dict.fromkeys(BASE, 1e9), [1.0]))
    seed = len(out)
    while len(out) < 100:
        # 나머지는 ion_flux만 바꾼 정상 변주 — 배치 전체 개수를 100으로 채운다.
        out.append((f"variant_{len(out)}", dict(BASE, ion_flux=float(10 + len(out) - seed)), [1.0]))
    return out[:100]


def main() -> int:
    report = []
    for name, recipe, times in cases():
        text = json.dumps({"recipes": [recipe], "times": times}, allow_nan=True)
        resp = client.post(
            "/predict", content=text,
            headers={"x-api-key": API_KEY, "content-type": "application/json"},
        )
        is_5xx = 500 <= resp.status_code < 600
        report.append({"name": name, "status": resp.status_code, "is_5xx": is_5xx})
        if is_5xx:
            print(f"[5xx!] {name}: {resp.status_code} {resp.text[:300]}")

    n_5xx = sum(1 for r in report if r["is_5xx"])
    print(f"total={len(report)} 5xx_count={n_5xx}")
    out_path = REPO_ROOT / "build" / "fuzz_predict_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"-> {out_path}")
    return 1 if n_5xx else 0


if __name__ == "__main__":
    sys.exit(main())
