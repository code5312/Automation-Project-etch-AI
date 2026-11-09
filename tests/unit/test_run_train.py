"""W4-01: 학습 스크립트 뼈대 단위 테스트(FR-06) — 시드 고정, config·git_sha·패키지 버전
저장이 기대대로 동작하는지 확인한다."""

import json
import pathlib
import random
import subprocess

import numpy as np
import torch
import yaml

from etchpipe.train.run_train import REPO_ROOT, git_sha, init_run, package_versions, set_seed


def test_set_seed_makes_random_draws_reproducible() -> None:
    set_seed(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_git_sha_matches_head() -> None:
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    sha = git_sha()
    assert sha.removesuffix("-dirty") == expected


def test_package_versions_reports_installed_packages() -> None:
    versions = package_versions(["numpy", "pyyaml", "this-package-does-not-exist"])
    assert versions["numpy"] != "not installed"
    assert versions["this-package-does-not-exist"] == "not installed"


def test_init_run_writes_config_git_sha_and_packages(tmp_path: pathlib.Path) -> None:
    config = {"seed": 7, "model": {"name": "meshgraphnet"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = init_run(config_path, out_root=tmp_path / "runs")

    assert run_dir.exists()
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == config
    assert len((run_dir / "git_sha").read_text().strip()) >= 40
    packages = json.loads((run_dir / "packages.json").read_text())
    assert packages["torch"] != "not installed"


def test_init_run_creates_distinct_run_dirs(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 1}))
    out_root = tmp_path / "runs"

    run_dir_a = init_run(config_path, out_root=out_root)
    run_dir_b = init_run(config_path, out_root=out_root)

    assert run_dir_a != run_dir_b
    assert run_dir_a.exists()
    assert run_dir_b.exists()
