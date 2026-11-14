# W8-08: 태그 v1.0, 불필요 파일 정리

## 1. 불필요 파일 정리

- `scripts/mgn_smoke_test.py` 삭제 — 1주차 스파이크 원본 그대로 남아 있던 죽은 코드.
  현재 ETL 출력 경로 규약(`data/processed/<dataset_hash>/graphs/<run_id>/`)과도
  안 맞아 지금 데이터로는 실행조차 안 됨. 같은 내용의 실제 증거는 이미
  `reports/evidence/W1-S3_rollout.py`/`W1-S3_rollout.md`로 따로 보존돼 있어 삭제해도
  증거 손실 없음(README가 이미 "정식 진입점 아님"으로 표시해 두고 있던 파일). README의
  해당 각주도 같이 제거.
- 그 외: 저장소 루트·`scripts/`에 추적 중인 캐시(`__pycache__`)·OS 파일·백업(`.bak`) 등
  불필요 파일 없음을 확인(`git ls-files` 전수 grep, 0건). `.gitignore`가 이미
  `__pycache__/`·`.pytest_cache/`·`.DS_Store` 등을 상시 차단 중.

## 2. 더 중요한 발견: 커밋이 "안 된" 재현성 공백

**[확인, 중요]** README가 "`runs/(mgn_tuned·fno_tuned 제외)`는 Git에서 제외됩니다"라고
이미 적어 뒀는데, 실제로는 `runs/fno_tuned/`가 **한 번도 커밋된 적이 없었다**
(`git ls-files runs/`에 `mgn_tuned/model.pt` 하나만 있음). `reports/eval/`도 전부
미추적. 영향:

- `scripts/run_model_comparison.py`(README에 문서화된 정식 진입점, `FNO_RUN_DIR`를
  직접 읽음)가 **깨끗한 clone에서 실행 불가**.
- `tests/unit/test_fno.py`는 자기 docstring에 이미 "로컬 전용이라 CI에서는 건너뛴다"고
  적어 둔 상태였다 — 즉 FNO 래퍼 코드는 지금까지 **CI에서 한 번도 실제로 테스트된 적이
  없었다**.
- 정의서 §4 목표5("깨끗한 환경에서 README만으로 전체 재현")와 정면으로 어긋남.

사용자에게 세 가지 선택지를 제시(①FNO 실제 체크포인트 커밋 ②결과 JSON만 커밋
③README만 정직하게 수정)하고, **①(실제 체크포인트 커밋)로 결정**(2026-11-14).

### 조치

- `.gitignore`에 `mgn_tuned`와 같은 패턴으로 `/runs/fno_tuned/` 예외 추가 — 재추론에
  꼭 필요한 4개 파일만: `config.json`, `inference.py`, `preprocess.py`,
  `model.safetensors`(teammate의 `metrics/`·`example/`·`model.pt`·README는 불필요해
  그대로 제외 — `inference.py`가 `model.safetensors`를 먼저 찾고 없을 때만 `model.pt`로
  후퇴하는 걸 코드로 확인).
- **부수 발견**: `runs/fno_tuned/inference.py`가 직접 `import safetensors`하는데
  `requirements-train.txt`에 없었다 — 지금까지 `nvidia-physicsnemo`의 전이 의존성으로
  조용히 설치돼 있어서 드러나지 않았을 뿐(`etch-train:0.1`에서 실측: `safetensors
  0.8.0`). `scikit-image`와 같은 이유로 명시 고정 추가(의존성 추가라 사용자 결정 필요한
  변경 — 같은 턴에서 실제 설치된 버전을 그대로 고정했고, 커밋 전 diff로 검토 가능).
- `tests/unit/test_fno.py`의 "로컬 전용/CI에서 건너뛴다" 설명을 "이제 CI에서 돈다"로
  갱신.
- README의 금지표현 검사 스크립트 설명이 `docs/data_card.md`/`docs/model_card.md`
  (W7-06/W8-03에서 추가)를 안 언급하고 있던 것도 같이 수정.

### 검증

```bash
$ docker run ... etch-train:0.1 pytest tests/unit/test_fno.py tests/unit/test_requirements_pinning.py -v
tests/unit/test_fno.py::test_load_fno_config_matches_known_grid PASSED
tests/unit/test_fno.py::test_rollout_and_curve_match_documented_example_depth PASSED
tests/unit/test_fno.py::test_predict_curve_from_vtp_returns_point_array PASSED
tests/unit/test_requirements_pinning.py::test_all_requirements_files_exist PASSED
tests/unit/test_requirements_pinning.py::test_all_dependencies_are_pinned_with_exact_version PASSED
5 passed in 30.78s

$ docker run ... etch-train:0.1 pytest tests/unit --ignore=test_serve.py --ignore=test_run_sim.py -q
217 passed, 3 skipped (+3개는 .git 마운트 안 한 로컬 아티팩트, 기존에도 알던 것과 동일)

$ python scripts/check_forbidden_expressions.py
금지 표현 검사 통과
```

전부 `etch-train:0.1`에서 실제 실행해 확인([확인], 추측 없음).

## 3. 태그 v1.0

태그는 git 작업이라 내가 직접 실행하지 않는다(표준 작업규칙). 위 변경을 전부 커밋한
뒤, 사용자가 다음을 실행하면 된다:

```bash
git tag -a v1.0 -m "etch-pipeline v1.0 — 8주차 완료, 증거①② 확보, 게이트 G7 충족"
git push origin v1.0
```

## 미확인/남은 작업

- 없음 — 발견한 재현성 공백은 이번에 바로 고쳤다(향후 과제로 미루지 않음, W7-08류
  "DOC-" 보류 대상과는 성격이 달라 바로 수정 가능한 범위였음).
