"""Run the CPU retraining policy once for a completed observation batch."""

import hashlib
import json
import math
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from etchpipe.registry.registry import Registry, sha256_file
from etchpipe.serve.inference import artifact_path_for


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _read_meta(raw: Path, rid: str) -> dict:
    return json.loads((raw / rid / "meta.json").read_text(encoding="utf-8"))


def _process_config(meta: dict) -> tuple[str, float | None]:
    config, rate = meta["sim_config_id"], meta["substrate_a_ie"]
    if not isinstance(config, str) or not config:
        raise ValueError("sim_config_id must name the observed process configuration")
    if rate is not None and (
        type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
    ):
        raise ValueError("substrate_a_ie must be null or a positive finite number")
    return config, rate


def _batch_identity(raw: Path, graphs: Path, eval_steps: tuple[int, ...]):
    ids = sorted(item.name for item in raw.iterdir() if item.is_dir())
    if len(ids) != 10:
        raise ValueError(f"Expected 10 observation recipes, found {len(ids)}")

    recipes = []
    configs = set()
    for rid in ids:
        meta = _read_meta(raw, rid)
        configs.add(_process_config(meta))
        last, interval = meta["actual_k"], meta["step_min"]
        if type(last) is not int or last < 1:
            raise ValueError(f"{rid}: actual_k must be a positive integer")
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError(f"{rid}: step_min must be positive and finite")
        steps = sorted({0, last, *(step for step in eval_steps if step <= last)})
        graph_hashes = [sha256_file(graphs / rid / f"g{step:03d}.npz") for step in steps]
        recipes.append((rid, meta["params_hash"], last, interval, graph_hashes))
    if len(configs) != 1:
        raise ValueError("All observation recipes must use the same process configuration")
    recipe_hashes = {recipe[1] for recipe in recipes}
    if len(recipe_hashes) != len(ids):
        raise ValueError("Observation batch contains duplicate recipes")
    config = configs.pop()
    digest = hashlib.sha256(json.dumps([config, recipes]).encode()).hexdigest()
    return digest, ids, recipe_hashes, config


def _prepare_training(training_raw: Path, job: Path, recipe_hashes: set, config: tuple):
    from etchpipe.retrain.execute import DriftState
    from etchpipe.sim.drift_recipes import drift_splits

    splits, _ = drift_splits()
    ids = [rid for group in splits.values() for rid in group]
    for rid in ids:
        meta = _read_meta(training_raw, rid)
        if _process_config(meta) != config:
            raise ValueError(f"{rid}: training data uses a different process configuration")
        if meta["params_hash"] in recipe_hashes:
            raise ValueError(f"{rid}: training or held-out recipe overlaps the observation batch")
    raw = job / "raw"
    for rid in ids:
        shutil.copytree(training_raw / rid, raw / rid)
    config_id, rate = config
    return DriftState(
        raw_dir=raw, processed_root=job / "processed", sim_config_id=config_id, substrate_a_ie=rate,
    )


def _run_cpu(state, checkpoint: Path, norm, job: Path) -> dict:
    from etchpipe.retrain.execute import make_fine_tune_fn, make_gate_fn, make_regenerate_fn
    from etchpipe.retrain.loop import run_retrain_loop

    attempts = []
    attempt_dir = job
    record: dict = {}

    def record_profiles(candidate: Path | None, profiles: dict) -> None:
        path = attempt_dir / "profiles.json" if candidate else job / "baseline.json"
        _write_json(path, profiles)

    tune = make_fine_tune_fn(state, norm, out_root=job / "checkpoints")
    gate = make_gate_fn(state, checkpoint, record_profiles=record_profiles)

    def tune_and_record(champion: Path, config: dict) -> Path:
        nonlocal attempt_dir, record
        attempt_dir = job / f"attempt-{config['attempt']}"
        attempt_dir.mkdir()
        record = {
            "config": config, "dataset_hash": state.dataset_hash,
            "gate_recipe_ids": state.eval_ids(),
        }
        try:
            candidate = tune(champion, config)
        except Exception as exc:
            _write_json(attempt_dir / "result.json", {**record, "error": str(exc)})
            raise
        record.update(candidate=str(candidate), candidate_sha256=sha256_file(candidate))
        _write_json(attempt_dir / "result.json", record)
        return candidate

    def gate_and_record(candidate: Path) -> dict:
        try:
            result = gate(candidate)
        except Exception as exc:
            _write_json(attempt_dir / "result.json", {**record, "gate_error": str(exc)})
            raise
        record["gate"] = result
        _write_json(attempt_dir / "result.json", record)
        attempts.append(record)
        return result

    result = run_retrain_loop(
        checkpoint, tune_and_record, gate_and_record, make_regenerate_fn(state),
    )
    return {"attempts": attempts, "passed_checkpoint": result["promoted_ckpt"]}


def _promote(registry, model_id, runs_root, batch_id, state, result) -> str | None:
    from etchpipe.eval.eval import safe_git_sha
    from etchpipe.gate.gate import GateResult
    from etchpipe.registry.registry import compute_valid_range

    # Distinct batch workers share one publication lock for this registry.
    with closing(sqlite3.connect(registry.root / ".drift-promotion.sqlite3")) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if registry.current_champion() != model_id:
            return None
        new_id = f"mgn_retrain_{batch_id}"
        target = artifact_path_for(new_id, runs_root)
        target.parent.mkdir(parents=True)
        shutil.copyfile(result["passed_checkpoint"], target)
        gate = result["attempts"][-1]["gate"]
        registered = registry.register(
            new_id, target, GateResult(gate["passed"], gate["exit_code"], gate["reasons"]),
            dataset_hash=state.dataset_hash, config={"source": "drift_batch", "batch_id": batch_id},
            git_sha=safe_git_sha(), metrics=gate["reference_metrics"],
            valid_range=compute_valid_range(state.raw_dir, state.splits["train"]),
        )
        if not registered:
            raise RuntimeError(f"Registry declined promotion of {new_id}")
        return new_id


def run_batch(
    intake: Path, *, training_raw: Path, registry_root: Path, runs_root: Path,
    promote: bool = False,
) -> dict:
    """Inspect a completed observation batch and run the existing CPU loop if it triggers."""
    from etchpipe.eval.eval import load_model_a
    from etchpipe.retrain.monitor import EVAL_STEPS, check_drift

    raw, graphs = intake / "raw", intake / "graphs"
    batch_id, ids, recipe_hashes, config = _batch_identity(raw, graphs, EVAL_STEPS)
    job = runs_root.resolve() / "retrain" / "batches" / batch_id
    summary = job / "job.json"
    try:
        job.mkdir(parents=True)
    except FileExistsError:
        previous = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {
            "batch_id": batch_id, "status": "running", "evidence_dir": str(job),
        }
        return {**previous, "duplicate": True}

    report = {
        "batch_id": batch_id, "status": "running", "evidence_dir": str(job),
        "observation_ids": ids, "process_config": config, "promotion_enabled": promote,
        "cd_source": "depth_proxy",
    }
    try:
        registry = Registry(registry_root)
        model_id = registry.current_champion()
        if not model_id:
            raise ValueError("No champion is registered")
        checkpoint = artifact_path_for(model_id, runs_root.resolve())
        checksum = sha256_file(checkpoint)
        if checksum != registry.load_card(model_id)["artifact_sha256"]:
            raise ValueError(f"Champion checkpoint hash mismatch: {model_id}")
        model, _, norm = load_model_a(checkpoint, device="cpu")
        report["champion"] = {"model_id": model_id, "sha256": checksum}
        drift = check_drift(model, norm, graphs, raw, ids)
        report["drift"] = drift
        _write_json(summary, report)
        if not drift["triggered"]:
            report["status"] = "no_drift"
        else:
            state = _prepare_training(training_raw, job, recipe_hashes, config)
            result = _run_cpu(state, checkpoint, norm, job)
            report["status"] = "blocked"
            report["attempts"] = len(result["attempts"])
            if result["passed_checkpoint"] is not None:
                report["status"] = "candidate_passed"
                if promote:
                    promoted = _promote(registry, model_id, runs_root, batch_id, state, result)
                    report["promoted_model_id"] = promoted
                    report["status"] = "promoted" if promoted else "champion_changed"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    _write_json(summary, report)
    return report
