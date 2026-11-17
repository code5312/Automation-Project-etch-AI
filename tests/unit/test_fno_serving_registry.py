from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from etchpipe.gate.gate import GateResult
from etchpipe.models.fno import FNOModel
from etchpipe.registry.artifacts import FNO_BUNDLE_FILES, resolve_artifact, verify_bundle
from etchpipe.registry.registry import Registry, sha256_file


def test_fno_bundle_integrity_and_concurrent_publication(tmp_path):
    registry = Registry(tmp_path / "registry")
    runs = tmp_path / "runs"
    base = tmp_path / "baseline.pt"
    base.write_bytes(b"baseline")
    common = {"dataset_hash": "fixture", "git_sha": "fixture", "metrics": {}, "valid_range": {}}
    gate = GateResult(True, 0)
    registry.register("initial", base, gate, config={}, **common)
    ready = Barrier(2)

    def publish(model_id):
        config = {"model_family": "fno"}
        artifact = resolve_artifact(model_id, {"config": config}, runs)
        artifact.parent.mkdir(parents=True)
        for filename in FNO_BUNDLE_FILES:
            artifact.with_name(filename).write_bytes(filename.encode())
        config["artifact_files_sha256"] = {
            name: sha256_file(artifact.with_name(name)) for name in FNO_BUNDLE_FILES
        }
        ready.wait(timeout=5)
        return registry.register(
            model_id, artifact, gate, config=config, expected_current="initial", **common
        )

    ids = ("candidate1", "candidate2")
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(publish, ids))
    assert results.count(True) == 1
    model_id = ids[results.index(True)]
    assert registry.current_champion() == model_id
    assert len(registry.history()) == 2
    assert not (registry.root / ids[results.index(False)]).exists()
    card = registry.load_card(model_id)
    artifact = resolve_artifact(model_id, card, runs)
    verify_bundle(registry, model_id, card, artifact)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(model_id, artifact, gate, config=card["config"], **common)
    artifact.with_name("config.json").write_text("tampered")
    with pytest.raises(ValueError, match="config.json"):
        verify_bundle(registry, model_id, card, artifact)
    with pytest.raises(ValueError, match="config.json"):
        registry.register("tampered", artifact, gate, config=card["config"], **common)
    assert registry.current_champion() == model_id


def test_fno_loads_implementation_from_each_artifact(tmp_path):
    for label in ("first", "second"):
        bundle = tmp_path / label
        bundle.mkdir()
        (bundle / "inference.py").write_text(
            f"def load(path, device=None): return object(), {{'version': {label!r}}}"
        )
        (bundle / "preprocess.py").write_text(f"def vtp_to_sdf(path, config): return {label!r}")
    first, second = (FNOModel(tmp_path / label) for label in ("first", "second"))
    assert first.cfg["version"] == first.sdf_from_vtp("t000.vtp") == "first"
    assert second.cfg["version"] == second.sdf_from_vtp("t000.vtp") == "second"
