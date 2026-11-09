# W3-06: 골든셋 확정

- 날짜: 2026-11-08
- 코드: `src/etchpipe/sim/sampling.py`(`normalize_params`, `select_golden_recipes` 신규),
  `scripts/make_golden.py`(신규)
- 테스트: `tests/unit/test_sampling.py`(선정 규칙 3건), `tests/unit/test_golden.py`(신규,
  골든셋 산출물 4건)

## 선정 규칙 [확인]

체크리스트·`configs/sim/base.yaml` 주석의 "중심5·경계3·극단2"를 다음으로 구체화:
1. SPACE(7차원, `sampling.py`)의 각 파라미터를 `[0,1]`로 정규화(log 스케일은 로그 공간에서).
2. 공간 중심 `(0.5,...,0.5)`까지 유클리드 거리로 정렬.
3. 가장 가까운 5개=중심, 가장 먼 2개=극단, 그 다음으로 먼 3개=경계.

train split(300건)에서만 선정([제안]: 골든셋은 "회귀 게이트, 고정"이 목적이라 일반화
평가(val/test/ood)와 성격이 다름 — 모델이 이미 본 레시피를 기준으로 삼아도 무방하다고
판단). 입력 recipes 순서가 같으면 항상 같은 10개가 나온다(결정적, seed 의존 없음 —
`data/recipes.json`/`splits.json`이 고정이므로 추가 seed 불필요).

## 실행 [확인]

```
PYTHONPATH=src python scripts/make_golden.py
golden/raw: 10건 복사, golden_hash=e95bdca4de55
  center r0185, r0007, r0240, r0079, r0254
  edge   r0235, r0100, r0015
  extreme r0274, r0187
```

`golden/raw/<run_id>/`(10개, `data/raw/<run_id>/`에서 복사, 재생성 아님)와
`golden/golden.json`(run_id·역할·params·`golden_hash`) 생성.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_sampling.py tests/unit/test_golden.py -q
15 passed

PYTHONPATH=src python -m pytest tests/unit -q
76 passed (기존 69 + 신규 7)

python -m ruff check src/etchpipe/sim/sampling.py tests/unit/test_sampling.py tests/unit/test_golden.py scripts/make_golden.py
All checks passed!
```

## "변경 시 테스트 실패" 수동 확인 [확인]

`golden/raw/r0007/meta.json`의 `params_hash`를 `"tampered00"`으로 바꾼 뒤 재실행:

```
AssertionError: assert '308e438b7819' == 'e95bdca4de55'
1 failed
```

원본으로 복구(`data/raw/r0007/meta.json`에서 재복사) 후 재실행 시 4 passed로 복귀 확인.
**주의 [확인]**: `compute_dataset_hash`(W3-04에서 이미 구현)는 `meta.json`에 저장된
`params_hash` 필드를 그대로 신뢰하고 합산하는 방식이라, `params_hash` 필드는 그대로 두고
`params` 값만 바꾸는 변조는 이 테스트로 잡히지 않는다(먼저 이 방식으로 시도했다가 직접
확인함). 정상적인 재생성·수정은 항상 `params_hash`까지 같이 갱신되므로(`run_sim.py`가
시뮬레이션 시점에 함께 씀) 실제 운영 시나리오(의도치 않은 재생성·파일 깨짐)는 이 테스트가
잡아낸다. `params_hash` 필드 자체를 건드리지 않는 수동 조작은 기존 `dataset_hash`와
동일한 한계이며 W3-06 범위를 넘는 별도 보강 과제로 남긴다.

## 영향 범위

- `golden/`는 `.gitignore`의 `/data/` 규칙(루트 전용)과 무관하고, `*.vtp` 전역 제외
  규칙에는 `!golden/**/*.vtp` 예외가 이미 있어(W1-02) 커밋 대상. `git status`로
  `golden/`가 무시되지 않고 untracked로 잡히는 것 확인.
- 게이트 코드(`src/etchpipe/gate/gate.py`)의 `check_golden_chamfer`/`golden_hash` 불일치
  차단 로직은 W2-07에서 이미 틀만 있고 더미 입력(`GOLDEN_PAIRS`)을 쓰는 상태 — 이번 golden
  raw 데이터를 실제 (입력, 정답) 그래프 쌍으로 ETL·연결하는 작업은 W5-05/W5-06(게이트 완성,
  부정 테스트) 범위로 남겨둔다 [미확인: 연결 시점].
