# W1-04 증거: pr-check 로컬 드라이런

- 날짜: 2026-10-27
- `.github/workflows/pr-check.yml`의 4개 잡(lint, unit-tests, viennaps-smoke, forbidden-expressions)을 실제 GitHub Actions에서 실행하지는 못했다(푸시하지 않음, 사용자 지시). 대신 각 잡이 실행할 명령을 **로컬 컨테이너에서 동일하게** 실행해 결과를 남겼다.

## lint + unit-tests (python:3.11-slim)

```
pip install -r requirements-dev.txt
pip install -e .
ruff check .
pytest tests/unit -v
```

```
--- ruff check ---
All checks passed!
--- pytest ---
collected 6 items
tests/unit/test_forbidden_expressions.py::test_detects_banned_phrase PASSED
tests/unit/test_forbidden_expressions.py::test_detects_multiple_distinct_phrases PASSED
tests/unit/test_forbidden_expressions.py::test_clean_text_has_no_violations PASSED
tests/unit/test_forbidden_expressions.py::test_docs_dir_is_not_scanned PASSED
tests/unit/test_requirements_pinning.py::test_all_requirements_files_exist PASSED
tests/unit/test_requirements_pinning.py::test_all_dependencies_are_pinned_with_exact_version PASSED
============================== 6 passed in 0.59s ===============================
```

## forbidden-expressions

```
python scripts/check_forbidden_expressions.py
```

1차 실행에서 실제로 위반을 잡아냄:

```
금지 표현 발견:
  README.md:3: '실제 fab 데이터'
```

→ README.md의 "실제 fab 데이터는 사용하지 않는다"는 부정문이지만 금지 문자열을 그대로 포함하므로 검사기가 규칙 그대로(문자열 존재 여부) 차단했다. "팹에서 수집한 데이터는 쓰지 않는다"로 문구를 바꿔 재검사:

```
금지 표현 검사 통과
exit: 0
```

## viennaps-smoke

별도 CI 잡으로 분리했지만 명령 자체는 W1-03의 `requirements-sim.txt` 컨테이너 테스트와 동일하다 — `reports/evidence/W1-03_sim_install.md` 참조(OS 패키지 `libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6` 설치 후 `import viennaps` 성공, 버전 4.5.0).

## 미확인

- 이 워크플로가 실제 GitHub Actions(ubuntu-latest) 러너에서도 동일하게 통과하는지는 PR을 올려야 확인된다. 로컬 컨테이너(Docker Desktop, WSL2)와 GitHub Actions 러너 환경 차이(예: apt 패키지 가용성)로 결과가 달라질 가능성이 있다 — 첫 PR에서 반드시 재확인해야 한다.
- 설계서 §8 CI 표의 `pr-check` 행은 "설정 스키마 검증"도 포함하지만, 아직 설정 스키마(`configs/`)가 비어 있어 이번 pr-check에는 넣지 않았다(이번 지시: 린트·단위테스트·viennaps 스모크·금지표현검사 4가지만). 설정 스키마 검증은 `configs/`가 채워지는 2주차 이후 추가해야 한다 — 문서(design §8)와 실제 workflow 범위가 일시적으로 어긋난 상태다.
