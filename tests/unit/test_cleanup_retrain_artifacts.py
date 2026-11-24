"""W9-01: 재학습 산출물 정리 스크립트 단위 테스트."""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from cleanup_retrain_artifacts import (  # noqa: E402
    find_attempt_dirs,
    find_orphan_drift_dirs,
    referenced_dataset_hashes,
)


def _mkdir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_find_attempt_dirs_only_matches_attempt_prefix(tmp_path: pathlib.Path) -> None:
    retrain_root = tmp_path / "retrain"
    _mkdir(retrain_root / "attempt1_111")
    _mkdir(retrain_root / "attempt2_222")
    _mkdir(retrain_root / "mgn_retrain_20261112T212130")  # 레지스트리 모델 디렉터리

    found = find_attempt_dirs(retrain_root)

    assert {p.name for p in found} == {"attempt1_111", "attempt2_222"}


def test_find_attempt_dirs_respects_min_age(tmp_path: pathlib.Path) -> None:
    retrain_root = tmp_path / "retrain"
    old_dir = _mkdir(retrain_root / "attempt1_111")
    new_dir = _mkdir(retrain_root / "attempt2_222")

    now = 10_000.0
    os.utime(old_dir, (now - 3600, now - 3600))  # 1시간 전
    os.utime(new_dir, (now - 60, now - 60))  # 1분 전

    found = find_attempt_dirs(retrain_root, min_age_seconds=1800, now=now)

    assert old_dir in found
    assert new_dir not in found


def test_find_attempt_dirs_empty_when_root_missing(tmp_path: pathlib.Path) -> None:
    assert find_attempt_dirs(tmp_path / "nope") == []


def test_referenced_dataset_hashes_reads_all_cards(tmp_path: pathlib.Path) -> None:
    registry_root = tmp_path / "registry"
    card_a = _mkdir(registry_root / "model_a")
    card_b = _mkdir(registry_root / "model_b")
    (card_a / "card.json").write_text(json.dumps({"dataset_hash": "hash_a"}))
    (card_b / "card.json").write_text(json.dumps({"dataset_hash": "hash_b"}))

    assert referenced_dataset_hashes(registry_root) == {"hash_a", "hash_b"}


def test_referenced_dataset_hashes_empty_when_root_missing(tmp_path: pathlib.Path) -> None:
    assert referenced_dataset_hashes(tmp_path / "nope") == set()


def test_find_orphan_drift_dirs_excludes_kept_hashes(tmp_path: pathlib.Path) -> None:
    drift_root = tmp_path / "processed"
    _mkdir(drift_root / "hash_live")
    _mkdir(drift_root / "hash_stale")

    orphans = find_orphan_drift_dirs(drift_root, keep_hashes={"hash_live"})

    assert [p.name for p in orphans] == ["hash_stale"]


def test_find_orphan_drift_dirs_empty_when_root_missing(tmp_path: pathlib.Path) -> None:
    assert find_orphan_drift_dirs(tmp_path / "nope", keep_hashes=set()) == []
