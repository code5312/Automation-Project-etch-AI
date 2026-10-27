"""requirements-*.txt가 모두 `==`로 버전이 고정되어 있는지 검사한다(CLAUDE.md: 의존성 버전 고정)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQ_FILES = [
    "requirements-sim.txt",
    "requirements-train.txt",
    "requirements-serve.txt",
    "requirements-dev.txt",
]


def _spec_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("--"):
            lines.append(line)
    return lines


def test_all_requirements_files_exist() -> None:
    for name in REQ_FILES:
        assert (ROOT / name).is_file(), f"{name} 없음"


def test_all_dependencies_are_pinned_with_exact_version() -> None:
    for name in REQ_FILES:
        path = ROOT / name
        for line in _spec_lines(path):
            assert "==" in line, f"{name}: 버전 미고정 - {line!r}"
