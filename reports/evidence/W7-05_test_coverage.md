# W7-05: 전체 테스트·커버리지 실행

프로젝트가 환경 3분리(dev/sim/train/serve)라 단일 환경에서 전체를 못 돈다. 실제로
모든 테스트를 커버할 수 있는 조합(`etch-train:0.1` + `etch-serve:0.1`)으로 나눠
돌리고, `coverage combine`으로 하나의 리포트로 합쳤다.

## 실행
```
# train 환경: test_serve.py(fastapi 필요)·test_run_sim.py(viennaps 필요) 제외 전부
docker run ... etch-train:0.1 pytest tests/unit -q \
  --ignore=tests/unit/test_serve.py --ignore=tests/unit/test_run_sim.py \
  --cov=etchpipe --cov-report=
-> 218 passed in 681s

# serve 환경: fastapi + viennaps 둘 다 있어서 나머지 두 파일을 여기서
docker run ... etch-serve:0.1 pytest tests/unit/test_serve.py tests/unit/test_run_sim.py \
  -q --cov=etchpipe --cov-report=
-> 36 passed in 157s

# 합산
coverage combine .coverage.train .coverage.serve
coverage report -m
```

**결과: 254 passed, 0 failed.** (`.coveragerc`에 컨테이너 경로(`/app/src/etchpipe`)
↔ 호스트 경로(`src/etchpipe`) 매핑을 추가해야 `combine`이 제대로 됨 — 안 하면
"No data to report"로 비어 보임, 실제로 겪어서 발견.)

## 커버리지: 전체 80.8% (1752줄 중 336줄 미실행)

| 모듈 | 커버리지 | 비고 |
|---|---|---|
| `etl.py` | 97.0% | 양호 |
| `eval/metrics.py` | 96.2% | 양호 |
| `gate/gate.py` | 96.2% | 양호 |
| `serve/app.py` | 95.8% | 양호 |
| `run_train.py` | 95.6% | 양호 |
| `sampling.py` | 93.3% | 양호 |
| `run_sim.py` | 88.7% | 양호 |
| `registry.py`/`loop.py`(train)/`monitor.py`/`drift_recipes.py`/`chamfer_loss.py`/`rollout.py`/`baseline.py`/`dummy.py`/`fno.py` | 100% | - |
| `eval/eval.py` | 53.7% | CLI `main()`·플롯 저장 경로 미실행(아래 참고) |
| `serve/inference.py` | 50.6% | `build_initial_graph`/`predict_recipe`(실제 ViennaPS 메시+롤아웃) 미실행 |
| `train_main.py` | 52.6% | CLI 학습 진입점 미실행 |
| `retrain/execute.py` | **28.4%**(이번에 0%→28.4%로 개선) | 아래 참고 |
| `models/meshgraphnet.py` | 0% | 실제 forward pass가 유닛테스트로 직접 안 불림 |
| `retrain/__main__.py` | 0% | `argparse` CLI 래퍼 |
| `sim/sim_batch.py` | 0% | 스파이크 코드(수정 금지, 사용자 지시) |

## `retrain/execute.py`가 낮은 이유 (0%→28.4%, 구조적 한계)

이 모듈은 실제 드리프트 데이터(`data/drift/`)·val 분할(`data/splits.json`)에 의존하는
통합 성격 코드다. 둘 다 `.gitignore` 대상이라 CI/클린 클론에는 없다 — 그래서 원래부터
"pytest로 커버"가 아니라 **실제 실행 + `reports/evidence/`**로 검증하는 전략을 써왔다
(`retrain_20261112T212130.json`, `ADR-0005_stage2_gate_rewrite.md`의 실제 스모크 결과
등 — 전부 이번 세션에서 실제 champion·실제 데이터로 돌려서 증거를 남김).

이번에 `tests/unit/test_execute.py`(신규, 로컬에 `data/drift/raw` 없으면 자동
skip)로 torch 없이도 되는 순수한 부분만 추가 커버:
- `DriftState()` 기본값이 실제 운영 경로와 같은 `dataset_hash=26a29dcca7da`를 내는지
  (회귀 테스트)
- `eval_ids()`가 `golden_drift ∪ val`인지
- `_val40_run_ids()`가 `data/splits.json`의 `val`과 일치하는지

`make_fine_tune_fn`/`make_gate_fn`/`run()`(나머지 71.6%)은 실제 torch 학습·실제
champion 비교·실제 레지스트리 승격이 얽혀 있어 의미 있게 모킹하면 실제 동작과
괴리가 커진다고 판단해 **pytest로 끌어올리지 않고 현재 전략(실제 실행 증거) 유지**
— 단, 이 테스트는 실제 ETL을 두 번 돌려 **8분 걸림**(느림, CI 부적합 — 로컬 전용
회귀 안전망으로만 의도).

## 조치
- `.coveragerc` 신규(컨테이너↔호스트 경로 매핑, 향후 재사용).
- `.gitignore`에 `.coverage`/`.coverage.*`/`htmlcov/` 추가(커버리지 산출물은
  기계 생성물이라 커밋 대상 아님).
- `tests/unit/test_execute.py` 신규(3개, 로컬 전용 skip 가드).

## 미확인 / 남은 작업
- `models/meshgraphnet.py`(실제 forward), `serve/inference.py`의 ViennaPS 경로,
  `eval.py`/`train_main.py`의 CLI 진입점은 여전히 pytest 커버리지가 낮음 — 전부
  실제 체크포인트·실제 ViennaPS가 필요해 가벼운 유닛테스트로 의미 있게 못 끌어올림,
  이번 범위에서는 현상 유지로 판단. 필요하면 별도 항목으로 분리해 통합 테스트(실제
  champion 기준)를 추가하는 방향 검토.
