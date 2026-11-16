import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from etchpipe.eval import eval as eval_module
from etchpipe.gate.gate import GateResult
from etchpipe.registry import registry as registry_module
from etchpipe.registry.registry import Registry, sha256_file
from etchpipe.retrain import batch, execute, monitor
from etchpipe.retrain.__main__ import main


@pytest.fixture
def setup_batch(tmp_path, monkeypatch):
    intake = tmp_path / "incoming"
    ids = [f"mon{i:04d}" for i in range(10)]
    for rid in ids:
        raw = intake / "raw" / rid
        graphs = intake / "graphs" / rid
        raw.mkdir(parents=True)
        graphs.mkdir(parents=True)
        (raw / "meta.json").write_text(json.dumps({
            "actual_k": 1, "step_min": 1.0, "sim_config_id": "live_test",
            "substrate_a_ie": 8.05, "params_hash": rid,
        }))
        for step in (0, 1):
            (graphs / f"g{step:03d}.npz").write_bytes(f"{rid}:{step}".encode())

    runs = tmp_path / "runs"
    registry = Registry(tmp_path / "registry")
    checkpoint = runs / "retrain" / "current_model" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"registered champion")
    registry.register(
        "current_model", checkpoint, GateResult(True, 0, []),
        dataset_hash="base", config={}, git_sha="test", metrics={}, valid_range={},
    )
    load = Mock(return_value=("current", "note", "norm"))
    monkeypatch.setattr(eval_module, "load_model_a", load)
    options = {
        "training_raw": intake / "training",
        "registry_root": registry.root,
        "runs_root": runs,
    }
    return SimpleNamespace(
        intake=intake, ids=ids, registry=registry, checkpoint=checkpoint,
        load=load, options=options,
    )


def test_cli_no_drift_and_copied_duplicate(setup_batch, monkeypatch, capsys):
    s = setup_batch
    failed_recipes = {"mon0009"}
    monkeypatch.setattr(monitor, "_recipe_eval", lambda model, norm, graphs, raw, rid, *args: {
        "depth_pred": 25.0 if rid in failed_recipes else 10.0,
        "depth_true": 10.0,
        "rollout_step_errors": [0.1],
    })
    train = Mock(side_effect=AssertionError("no training on a false trigger"))
    monkeypatch.setattr(batch, "_run_cpu", train)

    argv = [
        "retrain", "run-batch", str(s.intake),
        "--registry", str(s.registry.root), "--runs-root", str(s.options["runs_root"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "no_drift"
    assert result["drift"]["n_passed"] == 9
    assert result["champion"]["sha256"] == sha256_file(s.checkpoint)

    copied = shutil.copytree(s.intake, s.intake.with_name("copied"))
    monkeypatch.setattr(sys, "argv", [*argv[:2], str(copied), *argv[3:]])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["duplicate"] is True
    s.load.assert_called_once()
    assert s.load.call_args.args[0] == s.checkpoint
    train.assert_not_called()
    assert s.registry.current_champion() == "current_model"


def test_true_runs_existing_loop_and_records_all_attempts(setup_batch, monkeypatch):
    s = setup_batch
    monkeypatch.setattr(monitor, "_recipe_eval", lambda model, norm, graphs, raw, rid, *args: {
        "depth_pred": 25.0 if rid in {"mon0008", "mon0009"} else 10.0,
        "depth_true": 10.0,
        "rollout_step_errors": [0.1],
    })
    eval_ids = [f"drift{i:04d}" for i in range(40, 50)]
    state = SimpleNamespace(dataset_hash="first", splits={"train": ["train"], "val": eval_ids})
    state.eval_ids = lambda: eval_ids
    monkeypatch.setattr(batch, "_prepare_training", Mock(return_value=state))
    calls = []

    def fine_tune(champion, cfg):
        assert champion == s.checkpoint
        calls.append(cfg["attempt"])
        if cfg["attempt"] == 1:
            duplicate = batch.run_batch(s.intake, **s.options, promote=True)
            assert (duplicate["duplicate"], duplicate["status"]) == (True, "running")
        ckpt = s.options["runs_root"] / f"candidate{cfg['attempt']}" / "model.pt"
        ckpt.parent.mkdir()
        ckpt.write_bytes(str(cfg["attempt"]).encode())
        return ckpt

    def regenerate(n_train):
        assert n_train == 100
        state.dataset_hash = "expanded"

    def gate_for(state, champion, *, record_profiles):
        assert champion == s.checkpoint
        profiles = {
            rid: {
                "step": 1, "time_min": 1.0, "chamfer_by_step": {1: 0.2},
                "reference_xy_um": [[0, 0], [1, 0]], "prediction_xy_um": [[0, 0], [1, 0.2]],
            }
            for rid in eval_ids
        }
        record_profiles(None, profiles)

        def gate(ckpt):
            record_profiles(ckpt, profiles)
            passed = ckpt.read_bytes() == b"2"
            return {"passed": passed, "exit_code": 0 if passed else 1, "reasons": [],
                    "reference_metrics": {"gate_a": {"n": 20}, "gate_c": {}}}

        return gate

    monkeypatch.setattr(execute, "make_fine_tune_fn", lambda *args, **kwargs: fine_tune)
    monkeypatch.setattr(execute, "make_regenerate_fn", lambda *args: regenerate)
    monkeypatch.setattr(execute, "make_gate_fn", gate_for)
    monkeypatch.setattr(registry_module, "compute_valid_range", lambda *args: {})
    state.raw_dir = s.intake / "training"

    result = batch.run_batch(s.intake, **s.options, promote=True)
    directory = Path(result["evidence_dir"])
    assert result["drift"]["n_passed"] == 8
    assert calls == [1, 2]
    assert result["status"] == "promoted"
    assert s.registry.current_champion() == result["promoted_model_id"]
    assert json.loads((directory / "baseline.json").read_text()).keys() == set(eval_ids)
    attempts = [json.loads((directory / f"attempt-{i}" / "result.json").read_text())
                for i in calls]
    assert [record["dataset_hash"] for record in attempts] == ["first", "expanded"]
    assert [record["gate"]["passed"] for record in attempts] == [False, True]
    for i, attempt in zip(calls, attempts, strict=True):
        assert attempt["candidate_sha256"] == sha256_file(attempt["candidate"])
        profiles = json.loads((directory / f"attempt-{i}" / "profiles.json").read_text())
        assert profiles.keys() == set(eval_ids)
        assert profiles[eval_ids[0]]["chamfer_by_step"] == {"1": 0.2}
    assert batch.run_batch(s.intake, **s.options, promote=True)["duplicate"] is True
    assert calls == [1, 2]


def test_incomplete_batch_cannot_trigger_training(setup_batch, monkeypatch):
    s = setup_batch
    (s.intake / "graphs" / s.ids[-1] / "g001.npz").unlink()
    train = Mock()
    monkeypatch.setattr(batch, "_run_cpu", train)
    with pytest.raises(FileNotFoundError, match="g001.npz"):
        batch.run_batch(s.intake, **s.options)
    s.load.assert_not_called()
    train.assert_not_called()


def test_changed_champion_checkpoint_cannot_trigger_training(setup_batch, monkeypatch):
    s = setup_batch
    s.checkpoint.write_bytes(b"unexpected weights")
    train = Mock()
    monkeypatch.setattr(batch, "_run_cpu", train)
    result = batch.run_batch(s.intake, **s.options)
    assert result["status"] == "error"
    assert "Champion checkpoint hash mismatch" in result["error"]
    s.load.assert_not_called()
    train.assert_not_called()


def test_unchanged_failed_batch_is_not_automatically_retrained(setup_batch, monkeypatch):
    s = setup_batch
    monkeypatch.setattr(monitor, "check_drift", lambda *args, **kwargs: {"triggered": True})
    prepare = Mock(side_effect=FileNotFoundError("missing training assets"))
    train = Mock()
    monkeypatch.setattr(batch, "_prepare_training", prepare)
    monkeypatch.setattr(batch, "_run_cpu", train)
    first = batch.run_batch(s.intake, **s.options)
    duplicate = batch.run_batch(s.intake, **s.options)
    assert first["status"] == duplicate["status"] == "error"
    assert duplicate["duplicate"] is True
    assert "missing training assets" in first["error"]
    prepare.assert_called_once()
    train.assert_not_called()
    assert s.registry.current_champion() == "current_model"
