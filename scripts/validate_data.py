"""W3-05: raw 데이터 검증 스크립트.

모든 run에 대해 K+1(조기종료 시 actual_k+1) 스냅샷, NaN/Inf 없음, 점 수>0,
식각 깊이 단조 증가(비감소)를 확인하고, 점 수 최대치(그래프 크기 예산)를 보고한다.
사용: python scripts/validate_data.py [raw_dir] [splits.json]
  (기본 data/raw, data/raw 옆의 ../splits.json이 있으면 그 run_id 목록만 검사 —
  data/raw에는 1~2주차 스파이크 흔적(r0000_s1k2 등, actual_k 필드가 없는 옛 스키마)이
  섞여 있어 디렉터리 전체를 긁으면 본 데이터셋과 무관한 오류가 섞인다.)
종료 코드: 오류 있는 run이 하나라도 있으면 1.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from etchpipe.etl.etl import validate_dataset  # noqa: E402


def main() -> int:
    raw_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    splits_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else raw_dir.parent / "splits.json"
    run_ids = None
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        run_ids = sorted({rid for group in splits.values() for rid in group})
        print(f"splits.json 사용: {len(run_ids)}건 (스파이크 흔적 등 그 외 디렉터리는 제외)")
    report = validate_dataset(raw_dir, run_ids)
    print(f"{report['n_runs']} runs checked, {report['n_bad']} with errors")
    print(f"max_n_points (그래프 크기 예산) = {report['max_n_points']}")
    for r in report["bad_runs"]:
        print(" ", r["run_id"], r["errors"])
    return 1 if report["n_bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
