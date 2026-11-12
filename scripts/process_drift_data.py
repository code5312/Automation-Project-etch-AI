"""W6-03 2단계(`etch-train:0.1` 컨테이너 전용): `scripts/run_drift_data_gen.py`(sim
컨테이너)가 만들어 둔 `data/drift/raw/`를 ETL해 `data/drift/processed/<dataset_hash>/`에
graphs·stats.json·manifest.json·splits.json을 쓴다 — `etl.process_dataset` 그대로
재사용(기존 train 데이터 파이프라인과 같은 코드 경로)."""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.etl.etl import THETA_KEYS, process_dataset  # noqa: E402
from etchpipe.sim.drift_recipes import drift_splits  # noqa: E402

DRIFT_RAW_DIR = REPO_ROOT / "data" / "drift" / "raw"
DRIFT_PROCESSED_ROOT = REPO_ROOT / "data" / "drift" / "processed"


def main() -> int:
    splits, _ = drift_splits()
    manifest = process_dataset(DRIFT_RAW_DIR, splits, DRIFT_PROCESSED_ROOT, THETA_KEYS)
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
