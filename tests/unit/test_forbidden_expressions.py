"""scripts/check_forbidden_expressions.py 단위 테스트."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_forbidden_expressions.py"
_spec = importlib.util.spec_from_file_location("check_forbidden_expressions", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_detects_banned_phrase(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "demo.py").write_text(
        "# 실시간 추론을 지원한다\n", encoding="utf-8"
    )
    violations = _mod.find_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0][2] == "실시간"


def test_detects_multiple_distinct_phrases(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "demo.py").write_text(
        "# 정밀 시뮬레이션을 대체한다\n# 실제 fab 데이터로 검증했다\n",
        encoding="utf-8",
    )
    violations = _mod.find_violations(tmp_path)
    phrases = {v[2] for v in violations}
    assert phrases == {"대체한다", "실제 fab 데이터"}


def test_clean_text_has_no_violations(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "demo.py").write_text(
        "# 학습 범위 안에서만 유효한 추론 지연을 보고한다\n", encoding="utf-8"
    )
    assert _mod.find_violations(tmp_path) == []


def test_docs_dir_is_not_scanned(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "project-definition.md").write_text(
        "금지 표현: 대체한다, 실시간\n", encoding="utf-8"
    )
    assert _mod.find_violations(tmp_path) == []
