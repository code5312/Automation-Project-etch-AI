"""W9-01: 재학습 산출물 생애주기 정리.

지운다:
1. `runs/retrain/attempt*` 디렉터리 — `retrain/execute.py::run()`이 승격 시
   `model.pt`를 `Path.replace()`(이동)로 레지스트리 모델 디렉터리(`runs/retrain/<model_id>/`)
   로 옮기기 때문에, 승격 여부와 무관하게 루프가 끝난 뒤에는 항상 소비된 찌꺼기다
   (재학습 루프는 항상 원본 champion_ckpt에서 다시 시작하지 과거 attempt의 체크포인트를
   읽지 않음 — `retrain/loop.py::run_retrain_loop` 참조). 그래서 무조건 지워도 안전하다.
2. `data/drift/processed/<hash>/` 중 어떤 `registry/*/card.json`의 `dataset_hash`도
   가리키지 않는 디렉터리 — 과거 `DriftState.expand_train()` 호출로 생긴, 이제 아무
   모델도 참조하지 않는 중간/구버전 해시.

기본은 dry-run(목록만 출력). `--apply`를 줘야 실제로 지운다. 방금 생성된(진행 중일 수
있는) `attempt*` 디렉터리를 건드리지 않도록 `--min-age-minutes`(기본 30) 이전 것만
대상으로 한다.
"""

import argparse
import json
import pathlib
import shutil
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RETRAIN_RUNS_ROOT = REPO_ROOT / "runs" / "retrain"
DRIFT_PROCESSED_ROOT = REPO_ROOT / "data" / "drift" / "processed"
REGISTRY_ROOT = REPO_ROOT / "registry"
DEFAULT_MIN_AGE_MINUTES = 30


def find_attempt_dirs(
    retrain_root: pathlib.Path, min_age_seconds: float = 0.0, now: float | None = None
) -> list[pathlib.Path]:
    """`retrain_root` 바로 아래, 이름이 `attempt`로 시작하는 디렉터리 중 마지막 수정
    시각이 `min_age_seconds`보다 오래된 것만. 레지스트리 모델 디렉터리(`model_id`로
    이름 붙음)는 `attempt`로 시작하지 않으므로 애초에 안 걸린다."""
    if not retrain_root.exists():
        return []
    now = time.time() if now is None else now
    return sorted(
        p
        for p in retrain_root.iterdir()
        if p.is_dir()
        and p.name.startswith("attempt")
        and (now - p.stat().st_mtime) >= min_age_seconds
    )


def referenced_dataset_hashes(registry_root: pathlib.Path) -> set[str]:
    """레지스트리에 기록된(현재 챔피언이든 과거 모델이든) 모든 `dataset_hash`."""
    hashes: set[str] = set()
    if not registry_root.exists():
        return hashes
    for card_path in registry_root.glob("*/card.json"):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        dataset_hash = card.get("dataset_hash")
        if dataset_hash:
            hashes.add(dataset_hash)
    return hashes


def find_orphan_drift_dirs(
    drift_processed_root: pathlib.Path, keep_hashes: set[str]
) -> list[pathlib.Path]:
    """`drift_processed_root` 바로 아래, `keep_hashes`에 없는 해시 디렉터리."""
    if not drift_processed_root.exists():
        return []
    return sorted(
        p for p in drift_processed_root.iterdir() if p.is_dir() and p.name not in keep_hashes
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="실제로 삭제(기본은 dry-run)")
    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=DEFAULT_MIN_AGE_MINUTES,
        help=f"이보다 최근 수정된 attempt 디렉터리는 안 건드림(기본 {DEFAULT_MIN_AGE_MINUTES}분)",
    )
    args = ap.parse_args(argv)

    attempt_dirs = find_attempt_dirs(RETRAIN_RUNS_ROOT, min_age_seconds=args.min_age_minutes * 60)
    keep_hashes = referenced_dataset_hashes(REGISTRY_ROOT)
    drift_dirs = find_orphan_drift_dirs(DRIFT_PROCESSED_ROOT, keep_hashes)

    print(f"[runs/retrain/attempt*] {len(attempt_dirs)}건 ({args.min_age_minutes}분보다 오래됨)")
    for p in attempt_dirs:
        print(f"  - {p.relative_to(REPO_ROOT)}")
    print(f"[data/drift/processed/<고아 해시>] {len(drift_dirs)}건 (보존: {sorted(keep_hashes)})")
    for p in drift_dirs:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    if not args.apply:
        print("\n(dry-run - 실제 삭제는 --apply)")
        return 0

    for p in attempt_dirs + drift_dirs:
        shutil.rmtree(p)
    print(f"\n삭제 완료: attempt {len(attempt_dirs)}건 + drift {len(drift_dirs)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
