# W9-06: 드리프트 감지→재학습 자동 트리거 (ADR-0010)

## 1. 설계 중 발견한 두 가지 선행 문제

설계 착수 직후 두 가지 공백을 발견해 사용자에게 보고하고 먼저 처리했다(둘 다
ADR-0010에 같이 포함):

1. **champion 하드코딩**: `retrain/execute.py::CHAMPION_CKPT`·`scripts/
   run_drift_monitor.py::MGN_TUNED_CKPT`가 둘 다 `runs/mgn_tuned/model.pt`(최초
   baseline)로 고정돼 있어, 자동 재학습이 한 번이라도 승격되면 그 뒤로 영원히
   구버전만 보고 반복하는 문제. `serve/inference.py`는 W7-01에서 이미 같은 문제를
   고쳤지만 그 수정이 전파되지 않았었다.
2. **GitHub 호스팅 러너는 영속 디스크가 없음**: `registry/`·`runs/`·`data/`가 전부
   `.gitignore` 대상이라, 스케줄 워크플로우가 돌 때마다 빈 디스크에서 시작한다.
   사용자에게 지속 방식 3가지(GH Actions 캐시/self-hosted 러너/감지-알림만)를
   제시 — 질문을 두 번(사용자가 "다시 물어봐줘"로 재확인 요청) 거쳐 **GH Actions
   캐시** 선택 확정(2026-11-25, 두 차례 동일 답변).

## 2. 구현

### champion 동적 조회

- `registry/registry.py`: 공용 함수 `resolve_artifact_path(model_id, legacy_paths,
  retrain_runs_root)` 추가 — `serve/inference.py::_artifact_path_for`의 기존 로직을
  그대로 추출.
- `serve/inference.py`: `_artifact_path_for`가 위 함수로 위임하도록 교체 — 모듈
  전역(`_ARTIFACT_PATHS`/`_RETRAIN_RUNS_ROOT`)을 호출 시점에 그대로 읽어 넘겨 기존
  monkeypatch 테스트 7건 동작 불변.
- `retrain/execute.py`: `current_champion_ckpt()` 추가 — 레지스트리의 실제 현재
  champion을 조회, 없으면 `CHAMPION_CKPT`(baseline)로 후퇴, 아티팩트를 못 찾으면
  `RuntimeError`. `run()`이 `CHAMPION_CKPT` 상수 대신 이 함수를 쓰도록 교체(fine-tune
  시작점·게이트 비교 기준 둘 다).
- `scripts/run_drift_monitor.py`: `MGN_TUNED_CKPT` 상수 제거, `current_champion_ckpt()`
  로 교체. 결과 JSON에 `champion_ckpt` 필드 추가(어떤 champion을 체크했는지 추적
  가능하게).
- **범위 제외**: `run_retrain_demo.py`/`run_gate_negative.py`/`run_model_comparison.py`
  는 그대로(고정 기준점이 필요한 성격 — 데모·회귀 비교·벤치마크).

### 스케줄 자동 트리거

- `.github/workflows/drift-retrain-schedule.yml`(신규): 매주 월요일 03:00 UTC
  cron[제안, 임의 기본값] + `workflow_dispatch`(수동 테스트). `retrain-demo.yml`과
  같은 의존성 설치(torch/physicsnemo/ViennaPS 한 잡에 전부).
- `actions/cache@v4`로 `data/drift_monitor`를 `configs/sim/live.yaml` 해시 키로
  캐시(설정 안 바뀌면 수시간짜리 재시뮬레이션 생략).
- `actions/cache/restore`(접두사 복원) + `actions/cache/save`(매 실행 새 키,
  `if: always()`)로 `registry/`·`runs/retrain/`를 "누적 스냅샷"처럼 이어감.
- `reports/eval/W6-02_drift_monitor.json`의 `triggered`를 `jq`로 읽어 `true`일 때만
  `python -m etchpipe.retrain run` 실행, 결과를 워크플로우 아티팩트로 업로드.

## 3. 테스트

```
$ PYTHONPATH=src python -m pytest tests/unit/test_inference.py tests/unit/test_registry.py \
    tests/unit/test_execute_champion_resolution.py tests/unit/test_execute_fine_tune.py \
    tests/unit/test_monitor.py -q
31 passed in 20.57s
$ python -m ruff check .
All checks passed!
```

신규 테스트: `tests/unit/test_registry.py`에 `resolve_artifact_path` 3건(고정 매핑/
재학습 산출물 규칙/못 찾음), `tests/unit/test_execute_champion_resolution.py`(신규
파일) 4건(레지스트리 없음→후퇴, 레거시 model_id, 재학습 산출물, 아티팩트 없음→
RuntimeError) — 전부 torch 모델 로드 없이 레지스트리 레이어만 검증(`DriftState`
(`data/drift/raw` 필요)와 분리). 기존 `test_inference.py` 7건은 리팩터링 후에도
전부 그대로 통과(동작 불변 확인).

워크플로우 YAML 문법은 PyYAML로 파싱 확인(`on:` 키가 `True`로 뜨는 건 PyYAML 1.1의
알려진 표기 문제 — `retrain-demo.yml` 등 기존 워크플로우도 똑같이 나옴, 실제 GitHub
Actions 파서와는 무관함[확인]).

## 4. 미확인 / 다음 확인 필요

- **스케줄 워크플로우 자체의 실제 실행은 미확인**[미확인] — `workflow_dispatch`로
  수동 1회 실행해 (a) 캐시 적중/미스 분기, (b) `jq` 트리거 판정, (c) 실제
  `retrain.execute.run()` 연결이 의도대로 동작하는지 다음에 재확인 필요.
- **GH Actions 캐시의 알려진 한계**(ADR-0010 "명시적으로 인정하는 한계" 참고) —
  캐시 LRU 삭제, CI-로컬 레지스트리 분리, 콜드런 6시간 타임아웃 위험 — 전부 사용자가
  이미 인지하고 선택한 트레이드오프이며, 실제로 그 한계가 발현되는지는 아직 실측
  전[제안].
- `python -m etchpipe.retrain run` 실제 실행(실제 MeshGraphNet·실제 드리프트 데이터)
  은 W9-03/W9-04와 같은 이유로 로컬에서 재확인 못함(physicsnemo/torch_scatter 미설치,
  Docker 미기동) — CI(실제 설치됨) 또는 Docker 환경에서 재확인 필요.
