"""W7-05: `retrain/execute.py`는 실제 드리프트 데이터(`data/drift/`)·val 분할
(`data/splits.json`)에 의존하는 통합 성격 모듈이다 — 둘 다 `.gitignore` 대상이라
CI/클린 클론에는 없고, 로컬에서 실제 재학습을 한 번 돌려야 생긴다(이번 세션에서
실제로 그렇게 생성·검증함, `reports/evidence/retrain_20261112T212130.json` 등).
그래서 커버리지 측정에서 이 모듈이 0%로 나왔다 — 테스트가 없었던 게 아니라, 실제
실행 증거(reports/evidence/)로 검증하는 쪽을 택했기 때문[확인, 코드·데이터 상태
직접 확인]. 데이터가 로컬에 있을 때만 도는 가벼운 회귀 테스트를 추가해 최소한의
커버리지라도 남긴다 — `DriftState`/`_val40_run_ids`처럼 torch 없이도 되는 부분만."""

import json
import pathlib

import pytest

from etchpipe.retrain.execute import VAL40_SPLITS_PATH, DriftState, _val40_run_ids

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DRIFT_RAW_DIR = REPO_ROOT / "data" / "drift" / "raw"

pytestmark = pytest.mark.skipif(
    not DRIFT_RAW_DIR.exists(), reason="data/drift/raw 없음(로컬 전용, .gitignore 대상)"
)


def test_drift_state_default_args_match_known_dataset_hash() -> None:
    # W6-03/04 실제 운영 경로 그대로 — 이전에 실측 확인한 해시와 같아야 한다(회귀
    # 테스트, DriftState 리팩터링으로 기본 동작이 바뀌지 않았는지 확인).
    state = DriftState()
    assert state.dataset_hash == "26a29dcca7da"
    assert len(state.splits["train"]) == 40
    assert len(state.splits["val"]) == 10
    assert len(state.splits["golden_drift"]) == 10


def test_drift_state_eval_ids_is_golden_drift_plus_val() -> None:
    state = DriftState()
    expected = sorted(set(state.splits["golden_drift"]) | set(state.splits["val"]))
    assert state.eval_ids() == expected
    assert len(state.eval_ids()) == 20


@pytest.mark.skipif(not VAL40_SPLITS_PATH.exists(), reason="data/splits.json 없음(로컬 전용)")
def test_val40_run_ids_matches_splits_json() -> None:
    expected = json.loads(VAL40_SPLITS_PATH.read_text())["val"]
    assert _val40_run_ids() == expected
    assert len(_val40_run_ids()) == 40
