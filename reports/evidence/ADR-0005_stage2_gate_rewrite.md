# ADR-0005 2단계 증거: 게이트 집계 축 전환 + 드리프트 트리거 교체

2단계 범위(ADR-0005 결정 3·4(관문 C만, 관문 B 제외)·5): 레시피별 통과율 게이트,
드리프트 트리거 재학습 승격에 윌콕슨/HD95 회귀 검사, 드리프트 모니터 트리거 교체.

## 변경 파일
- `src/etchpipe/gate/gate.py`: `check_rollout_recipe`(레시피 1개, 비율 OR 증가량),
  `check_recipe_pass_rate`(결정3, 레시피별 3조건 + 통과율), `check_no_regression`
  (결정4 관문 C만, 윌콕슨 IoU + 최악 HD95 악화 + val40 재확인) 신설. 기존 `run_gate`는
  그대로 유지(W5-06 부정 테스트 등 기존 호출부 안 건드림).
- `src/etchpipe/train/rollout.py`: `rollout_val_error_by_step`에 `check_shape` 옵션
  추가(기본 False) — 스텝별 IoU·HD95 계산(Gate C①②에 필요). 기존 포지셔널 호출부
  (`rollout_val_error`) 순서 유지 확인(버그 1건 발견·수정 — 아래 참고).
- `src/etchpipe/retrain/monitor.py`: `check_drift`를 Chamfer 비율 방식에서
  `check_recipe_pass_rate` 방식으로 교체(결정5). `mean_chamfer_over_recipes`는 유지.
- `scripts/run_drift_monitor.py`: 골든셋 기준값 계산 단계 제거(더 이상 필요 없음),
  새 `check_drift` 시그니처로 갱신.
- `src/etchpipe/retrain/execute.py`: `make_gate_fn`을 Gate A(결정3) + 물리(ADR-0004
  결정4, 변경 없음) + Gate C(결정4, 관문 B 제외)로 재작성. 챔피언 드리프트셋
  IoU·HD95는 클로저 생성 시 1회만 계산(시도마다 재계산 안 함). Gate C③(val 40
  재확인)용 새 경로 상수(`VAL40_RAW_DIR`/`VAL40_GRAPHS_ROOT`/`VAL40_SPLITS_PATH`) 추가.
- `tests/unit/test_gate.py`, `test_monitor.py`, `test_rollout.py`: 신규 함수·옵션
  테스트 추가.

## 발견·수정한 버그
- **포지셔널 인자 밀림**: `rollout_val_error_by_step`에 `check_shape`를 기존
  `fix_mask`/`guard`/`guard_shrink`/`loop_removal` **사이에** 처음 넣었더니,
  `rollout_val_error`(얘만 순수 포지셔널 호출)의 `fix_mask`→`check_shape`,
  `guard`→`fix_mask`, `guard_shrink`→`guard`, `loop_removal`→`guard_shrink`로 전부
  한 칸씩 밀려 `TypeError`가 났다(실측: `test_rollout_val_error_*` 2건 실패,
  "'bool' object is not iterable"). 새 파라미터를 기존 파라미터 **뒤**(`loop_removal`
  다음)로 옮겨 수정 — 이후 다른 호출부(전부 키워드 인자 사용)는 애초에 영향 없었음을
  전수 확인.

## 실제 검증 (실제 챔피언 체크포인트 + 실제 데이터)
로컬 Python(torch 가능)으로 단위 테스트, `etch-train:0.1` 컨테이너로 실제
`make_gate_fn` 엔드투엔드 스모크(스크립트는 세션 스크래치패드, 커밋 안 함):

```
PYTHONPATH=src python -m pytest tests/unit/test_metrics.py tests/unit/test_gate.py \
  tests/unit/test_monitor.py tests/unit/test_rollout.py tests/unit/test_etl.py -q
-> 125 passed
ruff check src/ scripts/ tests/ -> All checks passed!
```

실제 champion(`runs/mgn_tuned/model.pt`) + 실제 드리프트 데이터(`data/drift/`,
60개 중 2개 샘플) + 실제 val(`data/raw`, `data/splits.json`, 2개 샘플)로
`make_gate_fn(state)(CHAMPION_CKPT)` 호출:
- Gate A만(기본 `recipe_pass_rate=0.9`): `passed=false`, 2개 중 1개만 통과(0.50<0.9,
  소표본이라 예상된 결과) — 정상적으로 레시피별 판정이 동작함을 확인.
- `recipe_pass_rate=0`로 강제(Gate A 통과시켜 Gate C 경로 검증): `passed=true`,
  후보=챔피언 자기 자신이라 `iou_wilcoxon_p=1.0`(완전 동일, allclose 분기)·
  `worst_hd95_degradation=0.0`·`val_pass_rate_ok=true` — Gate C③(val40 재확인)까지
  포함해 전체 경로가 예외 없이 실행되고 기대한 값(자기 자신과 비교 시 회귀 없음)을
  냄을 확인. 테스트 후 `gate.yaml` 원복 확인(byte-diff 동일).

## 미확인 / 남은 작업
- **실제 규모(20개 드리프트 평가셋 + 40개 val) 전체 실행은 안 함** — 스모크는 2개
  샘플로만 진행(시간 비용 때문, CPU 기준 레시피당 약 20초). 실제 재학습 루프 1회
  전체 실행은 이 2단계 작업과 별개로 사용자가 결정할 실행 환경(로컬 아님)에서
  진행 필요.
- CD는 여전히 깊이 값을 복제하는 기존 placeholder([미확인, 알려진 한계] —
  `gate.py`/`execute.py`/`monitor.py` 공통, 이번 ADR 범위 밖, 실제 CD 배선은
  별도 작업).
- `VAL40_GRAPHS_ROOT`의 `4b8c0ce1bc98` 해시는 `scripts/run_model_comparison.py`와
  같은 하드코딩 관례를 따름 — 데이터 재처리로 해시가 바뀌면 같이 갱신 필요.
- CI(GitHub Actions)에서 재확인 필요 — 로컬 Python/패키지 버전이 CI와 100% 동일하지
  않음.
