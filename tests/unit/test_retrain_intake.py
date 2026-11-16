import json
import sys
from unittest.mock import Mock

import pytest

from etchpipe.retrain import batch, intake
from etchpipe.retrain.__main__ import main


@pytest.fixture
def observation(tmp_path, monkeypatch):
    for index in range(10):
        recipe = tmp_path / "raw" / f"mon{index:04d}"
        snapshots = recipe / "snapshots"
        snapshots.mkdir(parents=True)
        (recipe / "meta.json").write_text(json.dumps({
            "actual_k": 1, "step_min": 1.0, "sim_config_id": "live_test",
            "substrate_a_ie": 8.05, "params_hash": recipe.name,
        }))
        for step in range(2):
            (snapshots / f"t{step:03d}.vtp").write_bytes(b"snapshot")

    def etl(source, out, keys):
        assert not (tmp_path / "graphs").exists()
        out.mkdir(parents=True)
        for step in range(2):
            (out / f"g{step:03d}.npz").write_bytes(f"{source.name}:{step}".encode())

    converter = Mock(side_effect=etl)
    monkeypatch.setattr(intake, "etl_run", converter)
    monkeypatch.setattr(intake, "validate_dataset", lambda *args: {"n_bad": 0})
    monkeypatch.setattr(sys, "argv", ["retrain", "etl-batch", str(tmp_path)])
    return tmp_path, converter


def test_handoff_publishes_all_graphs_then_dispatches_and_reuses_etl(
    observation, monkeypatch, capsys,
):
    root, converter = observation
    calls = []

    def run_batch(directory, **kwargs):
        assert directory == root
        assert len(list((directory / "graphs").glob("mon*/g001.npz"))) == 10
        marker = (directory / "graphs" / "ready.json").read_text()
        calls.append(marker)
        return {"status": "no_drift", "duplicate": len(calls) > 1}

    monkeypatch.setattr(batch, "run_batch", run_batch)
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["duplicate"] is False
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["duplicate"] is True
    assert len(calls) == 2 and calls[0] == calls[1]
    assert converter.call_count == 10


def test_etl_error_never_dispatches_and_can_retry(observation, monkeypatch, capsys):
    root, converter = observation
    etl = converter.side_effect

    def interrupt(source, destination, keys):
        if source.name == "mon0001":
            raise ValueError("unreadable snapshot")
        etl(source, destination, keys)

    converter.side_effect = interrupt
    run = Mock(return_value={"status": "no_drift"})
    monkeypatch.setattr(batch, "run_batch", run)
    assert main() == 1
    assert "unreadable snapshot" in capsys.readouterr().err
    assert not (root / "graphs").exists()
    assert not list(root.glob(".observation-etl-*"))
    run.assert_not_called()

    converter.side_effect = etl
    assert main() == 0
    run.assert_called_once()


def test_missing_frame_and_modified_published_batch_do_not_dispatch(
    observation, monkeypatch, capsys,
):
    root, _ = observation
    snapshot = root / "raw" / "mon0000" / "snapshots" / "t001.vtp"
    snapshot.unlink()
    run = Mock(return_value={"status": "no_drift"})
    monkeypatch.setattr(batch, "run_batch", run)
    assert main() == 1
    assert "snapshots do not match" in capsys.readouterr().err
    run.assert_not_called()

    snapshot.write_bytes(b"complete")
    assert main() == 0
    capsys.readouterr()
    snapshot.write_bytes(b"changed after ETL")
    assert main() == 1
    assert "Published observation batch changed" in capsys.readouterr().err
    run.assert_called_once()
