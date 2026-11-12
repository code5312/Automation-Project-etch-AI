"""C3: 학습 스크립트 뼈대(W4-01, FR-06).

YAML 설정 한 파일로 실행을 재현 가능하게 만든다: 시드 고정, 설정·git SHA·주요 패키지
버전을 `runs/<run_id>/`에 저장한다. 실제 MeshGraphNet 학습 루프(체크포인트·metrics.json)는
이 함수가 만든 디렉터리에 이어서 쓴다 — W4-02~04에서 추가.
"""

import json
import pathlib
import random
import subprocess
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

TRACKED_PACKAGES = [
    "torch", "torch_geometric", "torch_scatter", "nvidia-physicsnemo", "numpy", "scipy",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha(repo_root: pathlib.Path = REPO_ROOT) -> str:
    """현재 커밋 SHA. 커밋 안 된 변경이 있으면 "-dirty"를 붙인다(결과 재현성 경고용)."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    if status.strip():
        sha += "-dirty"
    return sha


def package_versions(names: list[str] = TRACKED_PACKAGES) -> dict[str, str]:
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def make_run_id() -> str:
    """사람이 읽을 수 있는 타임스탬프 + 짧은 난수 접미사(동일 마이크로초 충돌 방지)."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def init_run(
    config_path: str | pathlib.Path, out_root: str | pathlib.Path = "runs"
) -> pathlib.Path:
    """설정 로드 -> 시드 고정 -> runs/<run_id>/에 config.yaml·git_sha·packages.json 저장.
    반환한 디렉터리에 체크포인트·metrics.json을 이어서 쓴다(W4-02~04)."""
    config = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
    set_seed(config["seed"])

    run_dir = pathlib.Path(out_root) / make_run_id()
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "git_sha").write_text(git_sha(), encoding="utf-8")
    (run_dir / "packages.json").write_text(
        json.dumps(package_versions(), indent=2), encoding="utf-8"
    )
    return run_dir


if __name__ == "__main__":
    import sys

    print(init_run(sys.argv[1]))
