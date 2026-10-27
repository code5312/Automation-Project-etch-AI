"""W2-09: pipeline-smoke — 픽스처 → ETL → 더미 모델 → 평가 → 게이트 → 레지스트리 → 서빙 호출.

더미/최소판 전용이다. 실데이터·실모델이 아니라 tests/fixtures의 작은 데이터로
C1(완료분)~C7까지 "배선"이 끊기지 않는지 끝까지 확인하는 스모크 테스트.
"""

import json
import pathlib
import shutil

from fastapi.testclient import TestClient

from etchpipe.etl.etl import THETA_KEYS, process_dataset
from etchpipe.eval.metrics import chamfer_distance
from etchpipe.gate.gate import run_gate
from etchpipe.models.dummy import ZeroDisplacementModel
from etchpipe.registry.registry import Registry

FIXTURE_RAW = pathlib.Path("tests/fixtures/raw")
RUN_IDS = ["r0000", "r0001", "r0002"]
WORK = pathlib.Path("build/pipeline_smoke")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print("1) ETL: 픽스처(3 레시피) -> 그래프 + manifest")
    splits = {"train": RUN_IDS[:2], "val": RUN_IDS[2:], "test": [], "ood": []}
    manifest = process_dataset(FIXTURE_RAW, splits, WORK / "processed", THETA_KEYS)
    print("   dataset_hash:", manifest["dataset_hash"])

    print("2) 더미 모델 로드 (champion 역할: 변위 0)")
    model = ZeroDisplacementModel()

    print("3) 평가: 골든 역할로 레시피 전체의 t0->t1 전이쌍 사용")
    golden_pairs = []
    for rid in RUN_IDS:
        snaps = sorted((FIXTURE_RAW / rid / "snapshots").glob("t*.vtp"))
        from etchpipe.etl.etl import load_snapshot

        pos0, _, _, _ = load_snapshot(snaps[0])
        pos1, _, _, _ = load_snapshot(snaps[1])
        golden_pairs.append((pos0, pos1))
    errs = [chamfer_distance(model.predict(a), b) for a, b in golden_pairs]
    mean_chamfer = sum(errs) / len(errs)
    print(f"   zero-model vs t1 평균 Chamfer: {mean_chamfer:.4f} (스모크 참고치)")

    print("4) 게이트: 더미 임계값으로 PASS 확인(최소판 — 항목 ①만)")
    # 스모크 목적: 모델 성능이 아니라 "게이트가 호출되고 PASS 경로가 동작하는지"만 본다.
    gate_result = run_gate(model, golden_pairs, max_chamfer=mean_chamfer + 1e-6)
    print("   gate:", gate_result)
    if gate_result.exit_code != 0:
        print("   실패: 스모크는 PASS가 기대값인데 BLOCK됨")
        return 1

    print("5) 레지스트리: 승격 + history.jsonl")
    registry = Registry(WORK / "registry")
    registry.promote("dummy-smoke-v1", gate_result)
    print("   champion:", registry.current_champion())
    if registry.current_champion() != "dummy-smoke-v1":
        return 1

    print("6) 서빙: /health, /model, /predict 호출")
    import os

    os.environ.setdefault("ETCH_API_KEY", "smoke-test-key")
    from etchpipe.serve.app import app

    client = TestClient(app)
    key = os.environ["ETCH_API_KEY"]
    assert client.get("/health").status_code == 200
    assert client.get("/model", headers={"x-api-key": key}).status_code == 200
    recipe = {
        "ion_flux": 10.0, "etchant_flux": 5000.0, "oxygen_flux": 800.0,
        "ion_mean_energy": 100.0, "ion_exponent": 300.0, "mask_width": 10.0, "mask_height": 10.0,
    }
    resp = client.post(
        "/predict", json={"recipes": [recipe], "times": [0.5]}, headers={"x-api-key": key}
    )
    assert resp.status_code == 200

    summary = {
        "dataset_hash": manifest["dataset_hash"],
        "mean_chamfer": mean_chamfer,
        "gate_exit_code": gate_result.exit_code,
        "champion": registry.current_champion(),
        "serve_predict_status": resp.status_code,
    }
    json.dump(summary, open(WORK / "summary.json", "w"), indent=1)
    print("\npipeline-smoke 통과:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
