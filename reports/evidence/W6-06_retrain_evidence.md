# W6-06: 증거 ② 제출본 — 재학습 루프 무개입 완주

- 날짜: 2026-11-13
- 코드: `src/etchpipe/retrain/execute.py`(신설, `DriftState`/`fine_tune_fn`/`gate_fn`/
  `regenerate_fn`/`run`), `src/etchpipe/retrain/__main__.py`(신설, CLI),
  `scripts/compare_retrain_before_after.py`(신설, 전/후 비교)
- 실행 환경: `etch-serve:0.1`(torch+physicsnemo+ViennaPS 전부 있는 서빙 이미지를 그대로
  재사용 — 2차 시도 `regenerate_fn`이 ViennaPS도 필요해서, W6-08에서 "환경 3분리"를
  벗어난 것과 같은 이유)
- 실행 지시: 사용자 결정(2026-11-12) "CPU로 직접 돌려줘" — GPU 팀원 이관이나 GPU+DB
  아키텍처 변경 없이 기존 CPU/파일 기반 구조로 실제 실행

## 1단계: 드리프트 데이터 60개 생성 (W6-03) [확인]

```
PYTHONPATH=src python scripts/run_drift_data_gen.py (etch-sim:0.1, 백그라운드)
```

train 40 / val 10 / golden_drift 10, `substrate_a_ie=8.05`(×1.15 드리프트,
`configs/sim/live.yaml`). 레시피당 평균 약 10분(실측 범위 8~47분 안, 총 약 10시간 — 이
스크립트는 레시피를 순차 처리하는 구조라 코어 수와 무관, ViennaPS가 레시피 1개 안에서
이미 호스트 14코어를 거의 다 씀을 `docker stats`로 확인).

완전성 확인(60개 전부):

```
expected total: 60
missing: []
bad: []
splits sizes: {'train': 40, 'val': 10, 'golden_drift': 10}
```

## 2단계: 재학습 루프 실제 실행 (W6-04/05) [확인]

```
PYTHONPATH=src python -m etchpipe.retrain run (etch-serve:0.1)
```

**1차 시도에서 즉시 PASS** — 2~4차 에스컬레이션(ADR-0002) 불필요:

| 시도 | n_train | epochs(실제 사용, 하한) | lr_mult | 재생성 | 게이트 |
|---|---|---|---|---|---|
| 1 | 40 | 3 | 1.0 | 아니오 | **PASS**(exit_code=0) |

미세조정 손실(epoch별, `runs/retrain/attempt1_1794510586/metrics.json`):

| epoch | loss | 소요(초) |
|---|---|---|
| 0 | 0.02924 | 2517.0 |
| 1 | 0.02652 | 2502.9 |
| 2 | 0.02606 | 2405.1 |

CPU 3에폭 총 약 2시간(epoch당 약 40분, 40레시피분). 손실이 완만히 감소(0.0292→0.0261).

게이트 판정 근거(`reports/evidence/retrain_20261112T212130.json`, `golden_drift`+`val`
20개 평가): 롤아웃 후반/전반 Chamfer 비율 1.989(기준 ≤2.0), 선 꼬임 0건 → PASS.

## 3단계: 레지스트리 승격 [확인]

```
registry/current: mgn_tuned_ep2 -> mgn_retrain_20261112T212130
registry/history.jsonl 마지막 줄:
  {"action": "promote", "model_id": "mgn_retrain_20261112T212130", "promoted": true,
   "exit_code": 0, "reasons": [], "timestamp": "2026-11-12T21:21:30...+00:00"}
```

`registry/mgn_retrain_20261112T212130/card.json`·`valid_range.json` 생성 확인.

## 재학습 전/후 오차 표 [확인]

`gate.py::run_gate`의 `reference_metrics`는 ②(깊이/CD)가 **통과하면 수치를 남기지
않는다**(ADR-0004, 차단 전용 설계) — 그래서 실제 루프 실행(`retrain_20261112T212130.json`)
에는 깊이/CD 오차의 실제 숫자가 없다. `scripts/compare_retrain_before_after.py`로 같은
드리프트 평가셋(golden_drift+val 20개, `dataset_hash=26a29dcca7da`)에서 재학습 전
챔피언(`mgn_tuned_ep2`)과 재학습 후 모델을 독립적으로 재평가해 복원했다(게이트 판정을
다시 하는 게 아니라 보고용 — 롤아웃 설정은 `gate_fn`과 동일: `check_physics=True,
fix_mask=True, guard=True, loop_removal=True`).

| 지표 | 재학습 전(mgn_tuned_ep2) | 재학습 후(mgn_retrain) | 변화 |
|---|---|---|---|
| Chamfer 평균 | 0.5355 | 0.4979 | ↓ 7.0% |
| Chamfer 중앙값 | 0.5299 | 0.4659 | ↓ 12.1% |
| 깊이 %오차(평균) | **7.05%**(②기준 5% 초과) | **4.95%**(②기준 5% 이내) | BLOCK권 → PASS권 |
| 깊이/CD 차단 통과(②) | **false** | **true** | — |
| 롤아웃 후반/전반 비율(④) | 1.559 | 1.989 | 기준 2.0 안쪽 유지 |
| 선 꼬임(물리) | 0 | 0 | 변화 없음 |

재학습 후 "after" 수치(롤아웃 비율 1.989)는 실제 루프 실행의 게이트 판정값과 정확히
일치(`reference_metrics.rollout_late_early_ratio=1.9893788902927996`) — 독립 재평가와
실제 게이트 호출이 같은 값을 냄을 교차 확인.

**해석**: 드리프트(`substrate_a_ie`×1.15)를 맞은 기존 챔피언은 이 20개 평가셋 기준
②(깊이 오차 5% 이내)를 **실제로 못 넘긴다**(7.05%) — W6-02 모니터(10개 레시피, Chamfer
기준)에서는 안 걸렸던 성능 저하가, 이번 20개 드리프트 전용 평가셋·깊이 오차 기준으로는
실제로 드러남. 재학습(3에폭 미세조정)으로 깊이 오차가 5% 밑으로 내려와 ②를 통과하는
수준까지 회복됨 — 재학습 루프가 실제로 "막힐 뻔한 모델을 통과 가능한 수준으로 되돌리는"
기능을 함을 실측으로 확인했다.

**주의 [미확인]**: `cd_pred`/`cd_true`는 `gate_fn`과 동일하게 깊이 값을 그대로 복제해
쓴 것(CD 미배선, 알려진 한계 — `execute.py` 주석 참고)이라, 위 "깊이/CD" 판정은 실질적으로
깊이만 반영한다.

## 테스트/검증

```
python -m ruff check src/etchpipe/retrain/ scripts/compare_retrain_before_after.py
  → All checks passed!
PYTHONPATH=src python -c 'from etchpipe.retrain.execute import run, DriftState, ...'
  → import ok (etch-serve:0.1)
```

실제 end-to-end 실행(위 1~3단계)이 그 자체로 검증 — ADR-0002가 요구하는 "사람 개입 없이
자동 반복(최대 4회) 포함 완주" 정의를 충족(이번엔 1차에서 종료돼 2~4차 경로는 코드만
존재, `tests/unit/test_retrain_loop.py`의 모킹 테스트로 이미 검증됨, 실제 미세조정 데이터로는
미실행[미확인]).
