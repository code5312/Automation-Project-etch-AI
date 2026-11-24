# W9-03: 재학습 루프 베스트 체크포인트 선택 (ADR-0006)

## 1. 설계

ADR: `docs/adr/0006-retrain-loop-best-checkpoint-selection.md`. 핵심: 2026-11-17 진단
(attempt 3에서 성능 악화)의 원인 중 하나였던 "마지막 에폭 가중치를 무조건 저장"을,
`train_main.py::run_training()`에 이미 있는 검증 손실 기반 선택 패턴으로 교체.

## 2. 구현

`src/etchpipe/retrain/execute.py::make_fine_tune_fn`의 `fine_tune_fn` 수정:
- 매 에폭 `train_one_epoch` 뒤 `state.splits["val"]`(기존에 fine-tune 쪽에서는 안 쓰던
  분할, 게이트 평가에서만 간접 사용되던 걸 재사용)로 `evaluate()` 호출.
- 지금까지 최저 검증 손실의 `state_dict`를 메모리에 보관, 루프 끝나면 그걸 저장(마지막
  에폭이 아니라).
- `val_ids`가 비면(방어적 처리, 설계상 안 비지만) 마지막 에폭으로 후퇴 — 기존 동작 유지.
- `metrics.json` 포맷이 평평한 리스트에서 `{"epochs": [...], "best_epoch":, "best_val_loss":}`
  로 바뀜 — 전체 코드베이스 grep 확인(`grep -rn "metrics.json" src/etchpipe/retrain/
  scripts/`) 결과 이 파일을 다시 읽는 코드가 없어[확인, 2026-11-24] 하위호환 걱정 없이
  바꿈.

`fine_tune_fn`의 외부 시그니처(`(champion_ckpt, cfg) -> ckpt_path`)는 그대로라
`retrain/loop.py::run_retrain_loop`·`run()`·`run_retrain_demo.py` 쪽 변경 불필요.

## 3. 테스트

`tests/unit/test_execute_fine_tune.py` 신규 2건 — `load_model_a`/`train_one_epoch`/
`evaluate`를 전부 더미로 몽키패치(physicsnemo/torch_scatter 불필요, `PYTHONPATH=src`만
있으면 로컬에서 바로 실행 가능 — 둘 다 로컬에 미설치 확인됨, 2026-11-24):

1. `test_fine_tune_fn_saves_best_val_epoch_not_last` — 검증 손실 순서 `[0.5, 0.3, 0.6,
   0.4]`(최저가 epoch 1, 마지막 epoch 3이 아님)를 주고, 저장된 체크포인트가 epoch 1의
   state임을 확인.
2. `test_fine_tune_fn_falls_back_to_last_epoch_when_no_val` — `val` 분할이 비면 기존
   동작(마지막 에폭 그대로 저장)으로 후퇴하는지 확인.

```
$ PYTHONPATH=src python -m pytest tests/unit/test_execute_fine_tune.py -v
2 passed in 13.13s
$ python -m ruff check src/etchpipe/retrain/execute.py tests/unit/test_execute_fine_tune.py
All checks passed!
```

## 4. 미확인 / 다음 확인 필요

- **실제 MeshGraphNet·실제 드리프트 데이터로의 통합 실행은 미확인**[미확인] — 로컬에
  physicsnemo/torch_scatter가 설치돼 있지 않고(`ModuleNotFoundError` 확인) Docker
  데몬도 현재 미기동이라 `python -m etchpipe.retrain run`을 실제로 돌려 재확인하지
  못했다. `tests/unit/test_execute.py`가 이미 선언한 것과 같은 방침(이 모듈은 실제
  실행 증거로 검증) — Docker 환경이 되는 대로 실제 1회 재학습을 돌려 `metrics.json`에
  `best_epoch`/`val_loss`가 정상 기록되는지 재확인 필요.
- 매 에폭 val 평가 추가에 따른 실제 시간 오버헤드도 미실측[미확인, ADR-0006 "비용·
  트레이드오프" 절 참고] — 위 재확인 때 같이 잴 것.
- 데모(`build/retrain_demo/`)로 2026-11-17 진단(attempt 3 악화)이 이 수정으로 실제
  해소되는지는 별도로 `run_retrain_demo.py`를 다시 돌려야 확인 가능 — 이번 항목
  범위에는 포함하지 않음(다음 단계로 제안).
