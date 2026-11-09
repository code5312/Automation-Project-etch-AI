"""W3-06: 골든셋(golden/raw) 회귀 테스트.

golden/golden.json에 기록된 golden_hash를 golden/raw의 실제 meta.json에서 다시
계산해 비교한다. golden/raw의 파일이 하나라도 바뀌면(params_hash가 바뀌므로)
이 테스트가 실패한다 — "변경 시 테스트 실패"(체크리스트 W3-06) 요건.
"""

import json
import pathlib

import pytest

from etchpipe.etl.etl import compute_dataset_hash, validate_dataset

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN_JSON.exists(), reason="golden/golden.json 없음(scripts/make_golden.py 실행 전)"
)


def _manifest() -> dict:
    return json.loads(GOLDEN_JSON.read_text())


def test_golden_set_has_10_runs() -> None:
    m = _manifest()
    assert len(m["run_ids"]) == 10
    assert sorted(d.name for d in (GOLDEN_DIR / "raw").iterdir()) == sorted(m["run_ids"])


def test_golden_set_role_counts() -> None:
    m = _manifest()
    roles = [s["role"] for s in m["selected"]]
    assert roles.count("center") == 5
    assert roles.count("edge") == 3
    assert roles.count("extreme") == 2


def test_golden_hash_matches_recorded_value() -> None:
    """golden/raw가 변경되면(의도치 않은 수정·재생성) 이 해시가 달라져 실패한다."""
    m = _manifest()
    recomputed = compute_dataset_hash(m["run_ids"], GOLDEN_DIR / "raw")
    assert recomputed == m["golden_hash"]


def test_golden_raw_passes_validation() -> None:
    m = _manifest()
    report = validate_dataset(GOLDEN_DIR / "raw", m["run_ids"])
    assert report["n_bad"] == 0, report["bad_runs"]
