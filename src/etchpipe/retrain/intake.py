"""Prepare and publish one complete observation batch for the drift monitor."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from etchpipe.etl.etl import THETA_KEYS, etl_run, validate_dataset
from etchpipe.registry.registry import sha256_file
from etchpipe.retrain.batch import _batch_identity
from etchpipe.retrain.monitor import EVAL_STEPS


def _raw_identity(raw: Path) -> tuple[list[str], str]:
    ids = sorted(path.name for path in raw.iterdir() if path.is_dir())
    if len(ids) != 10:
        raise ValueError(f"Expected 10 observation recipes, found {len(ids)}")

    entries = []
    for rid in ids:
        recipe = raw / rid
        metadata = recipe / "meta.json"
        last = json.loads(metadata.read_bytes())["actual_k"]
        if type(last) is not int or last < 1:
            raise ValueError(f"{rid}: actual_k must be a positive integer")
        snapshots = sorted((recipe / "snapshots").glob("t*.vtp"))
        if [path.name for path in snapshots] != [f"t{step:03d}.vtp" for step in range(last + 1)]:
            raise ValueError(f"{rid}: snapshots do not match t000 through t{last:03d}")
        entries.append((rid, sha256_file(metadata), [sha256_file(path) for path in snapshots]))
    return ids, hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _check_published(raw: Path, graphs: Path, raw_digest: str) -> None:
    marker = graphs / "ready.json"
    if not marker.is_file():
        raise ValueError(f"{graphs}: existing graphs have no ETL completion record")
    expected = json.loads(marker.read_text(encoding="utf-8"))
    batch_id, _, _, _ = _batch_identity(raw, graphs, EVAL_STEPS)
    if expected != {"raw_sha256": raw_digest, "batch_id": batch_id}:
        raise ValueError("Published observation batch changed; use a new intake directory")


def prepare_observation(intake: Path) -> None:
    """Publish the graph directory atomically after all ten recipes finish ETL."""
    raw, graphs = intake / "raw", intake / "graphs"
    ids, raw_digest = _raw_identity(raw)
    if graphs.exists():
        _check_published(raw, graphs, raw_digest)
        return

    validation = validate_dataset(raw, ids)
    if validation["n_bad"]:
        errors = "; ".join(
            f"{recipe['run_id']}: {', '.join(recipe['errors'])}"
            for recipe in validation["bad_runs"]
        )
        raise ValueError(f"Invalid raw observation snapshots: {errors}")

    with TemporaryDirectory(dir=intake, prefix=".observation-etl-") as temporary:
        staged = Path(temporary) / "graphs"
        for rid in ids:
            etl_run(raw / rid, staged / rid, THETA_KEYS)
        if _raw_identity(raw) != (ids, raw_digest):
            raise ValueError("Raw observation batch changed during ETL")
        batch_id, _, _, _ = _batch_identity(raw, staged, EVAL_STEPS)
        (staged / "ready.json").write_text(
            json.dumps({"raw_sha256": raw_digest, "batch_id": batch_id}), encoding="utf-8",
        )
        try:
            staged.rename(graphs)
        except OSError:
            if not graphs.exists():
                raise
            _check_published(raw, graphs, raw_digest)
