"""금지 표현 검사: CLAUDE.md의 금지 표현이 코드·README·대시보드·모델 카드에 쓰였는지 확인한다.

사용: python scripts/check_forbidden_expressions.py
종료 코드: 0(위반 없음) / 1(위반 발견)
"""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN = [
    "대체한다",
    "실시간",
    "실제 fab 데이터",
    "SK하이닉스 모델 재현",
    "어떤 AI가 효율적인지 입증",
]

# 금지 표현 규칙(CLAUDE.md)의 적용 범위: 코드·README·대시보드·모델 카드.
# docs/의 설계 문서(project-definition.md/project-design.md/adr/*)와 CLAUDE.md 자신은
# 금지 표현을 "정의"하거나 무관한 맥락(예: "실시간", "~을 대체한다"의 일반 단어 용법)으로
# 포함하므로 검사 대상에서 뺀다. 체크리스트(W7-06)가 명시한 "모델 카드"는 데이터 카드와
# 같은 종류의 실제 문서이므로 개별 추가한다.
# docs/model_card.md는 W8-03에서 생성됨과 동시에 추가함(2026-11-14).
SCAN_DIRS = ["src", "scripts", "tests"]
SCAN_FILES = ["README.md", "docs/data_card.md", "docs/model_card.md"]

# 이 검사 스크립트와 그 테스트는 금지 표현을 문자열 리터럴로 포함하므로 자기 자신은 제외한다.
SELF_EXCLUDE = {
    Path("scripts/check_forbidden_expressions.py"),
    Path("tests/unit/test_forbidden_expressions.py"),
}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".html", ".js", ".ts", ".tsx"}


def iter_target_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = root / d
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES)
    for f in SCAN_FILES:
        p = root / f
        if p.is_file():
            files.append(p)
    return [p for p in files if p.relative_to(root) not in SELF_EXCLUDE]


def find_violations(root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in iter_target_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for phrase in FORBIDDEN:
                if phrase in line:
                    violations.append((path, lineno, phrase))
    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations = find_violations(root)
    if violations:
        print("금지 표현 발견:")
        for path, lineno, phrase in violations:
            print(f"  {path.relative_to(root)}:{lineno}: '{phrase}'")
        return 1
    print("금지 표현 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
