# ADR-0005 1단계 증거: 평가 지표 신설·재정의

1단계 범위(ADR-0005 결정 1·2·2-1·6): 신규 지표(표면 Dice, HD95, 추론 속도 비), 기존
지표 재정의(IoU, CD 측정 높이, 테이퍼 분모, 보잉), `gate.yaml` 값 갱신.

## 변경 파일
- `src/etchpipe/eval/metrics.py`: `surface_dice`/`hd95`/`speed_ratio`/
  `_restrict_to_silicon` 신설, `chamfer_distance`에 `silicon_only`(기본 False, 기존
  호출부 수치 보존) 추가, `iou`/`cd_profile`/`taper_angle_deg`/`bowing` 재정의
- `tests/unit/test_metrics.py`: 변경된 함수 테스트 갱신 + 신규 함수 테스트 추가
- `configs/pipeline/gate.yaml`: `max_chamfer` 0.582→0.3, `rollout_max_ratio` 2.0→2.5,
  2단계 전용 키(`cd_floor`/`recipe_pass_rate`/`rollout_min_increase`/`alpha_retrain`)
  미리 선언(아직 `gate.py` 미사용)
- `docs/project-definition.md` §9, `docs/project-design.md` C4: 지표 목록 갱신

## 영향 범위 확인
- `chamfer_distance` 실제 호출부 5곳(`gate.py`, `train/rollout.py`,
  `scripts/run_baselines.py`, `scripts/run_model_comparison.py`,
  `scripts/run_pipeline_smoke.py`) 전부 2개 위치 인자만 써서 `silicon_only` 기본값
  영향 없음 — 기존 수치 불변[확인, 전수 grep].
- `iou()`는 `tests/unit/test_metrics.py` 외 실제 호출부 없음(gate.py 등 어디서도 안 씀)
  — 시그니처 변경(3번째 인자 `mask_initial` 추가)의 실질 영향 없음[확인].
- `cd_profile`/`taper_angle_deg`는 `src/etchpipe/serve/inference.py`가 호출하지만
  시그니처는 그대로(내부 계산만 바뀜) — `/predict` 응답의 cd/taper 수치가 바뀌는 건
  의도된 결과(ADR-0005 결정2).
- `bowing()`은 실제 호출부 없음(테스트만) — 시그니처를 스칼라 3개에서
  `(mask, xr, yr, depth_value)`로 바꿔도 영향 없음.

## 증거
```
PYTHONPATH=src python -m pytest tests/unit/test_metrics.py tests/unit/test_gate.py tests/unit/test_etl.py tests/unit/test_monitor.py -q
-> 94 passed
python -m ruff check src/etchpipe/eval/metrics.py tests/unit/test_metrics.py
-> All checks passed!
```
(로컬 Python 3.x, numpy/pytest/ruff — torch/ViennaPS 불필요한 순수 함수라 로컬에서
직접 실행·확인함. `test_eval.py`/`test_rollout.py`/`test_serve.py` 등 torch·fastapi가
필요한 테스트는 이번 변경 대상이 아니라 별도 확인 안 함.)

## 미확인 / 남은 작업
- CI(etch-train:0.1/etch-serve:0.1 이미지, GitHub Actions)에서 재확인 필요 — 로컬
  Python 버전·패키지 버전이 CI와 100% 동일하지 않음.
- `inference.py`의 `/predict` 응답 cd/taper 수치가 실제로 바뀐 걸 실제 champion
  체크포인트로 확인하는 건 안 함(torch 필요, 1단계 범위 밖).
- ADR-0005 2단계(게이트 집계 축 전환, 윌콕슨 검정, 드리프트 트리거 교체)는 아직 시작
  안 함 — `gate.yaml`의 2단계 전용 키는 선언만 돼 있고 `gate.py`가 아직 안 읽음.
