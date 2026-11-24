# ADR-0006: 자동 재학습 루프에 검증 손실 기반 베스트 체크포인트 선택 추가

- 날짜: 2026-11-24
- 상태: **승인(사용자 결정 2026-11-23, W9-03 진행 지시)**
- CLAUDE.md 작업규칙 7: 근거를 보여주고 사용자 확인을 받은 뒤 다음 단계로 간다 — 이 ADR은
  2026-11-17 진단(아래 배경)과 2026-11-23 사용자 승인(v1.1 체크리스트 작성·W9-03 지시)을
  근거로 한다.
- 영향 파일: `src/etchpipe/retrain/execute.py::make_fine_tune_fn`,
  `tests/unit/test_execute_fine_tune.py`(신규), `docs/project-checklist.md` W9-03.
- **범위 제외**: 재학습 시도 간 누적 학습 여부(매 시도가 원본 `champion_ckpt`에서
  재시작하는 것 자체)는 이 ADR의 범위가 아니다 — 별도 결정(W9-05)으로 남긴다.

## 배경

2026-11-17 사용자 질문("왜 attempt 3에서 성능이 나빠지냐")에 대한 코드 조사로 두 가지를
확인했다:

1. `retrain/execute.py::make_fine_tune_fn`의 `fine_tune_fn`은 지정된 에폭 수만큼 돌고
   **마지막 에폭의 가중치를 무조건 저장**한다 — 검증 손실 기반 선택이 없다.
2. 반면 `train/train_main.py::run_training()`은 이미 같은 문제(과적합으로 마지막 에폭이
   최선이 아닐 수 있음, 2026-11-10 `reports/model_issues_mgn.md` 문제①)를
   `val_ids`가 있으면 매 에폭 `evaluate()`로 검증 손실을 계산해 최저값일 때
   `ckpt_best.pt`를 따로 저장하는 방식으로 이미 해결해 두었다.

데모(`build/retrain_demo/result.json`)에서 attempt 3(에폭 10~15, 레시피 5개)의 깊이오차가
attempt 1(에폭 3~5)보다 악화(11.5%→22.9%)된 현상이, 이 둘(①매 시도 원본에서 재시작·②베스트
체크포인트 선택 없음)의 조합으로 설명된다고 진단했다 — 작은 데이터셋에서 에폭을 늘릴수록
과적합 보호 장치 없이 마지막 에폭을 그대로 쓰는 게 구조적 위험이라는 결론.

## 결정

`train_main.py::run_training()`의 패턴을 `fine_tune_fn`에 포팅한다:

- `DriftState`는 이미 `state.splits["val"]`(기본 10개 레시피)을 들고 있지만 `fine_tune_fn`은
  지금까지 이걸 전혀 쓰지 않았다(게이트 평가에서만 `state.eval_ids()`로 간접 사용) — 이
  val 분할을 그대로 재사용한다. 새 데이터 분할이나 설정 추가 불필요.
- 매 에폭 `train_one_epoch` 뒤 `evaluate(model, state.graphs_root, state.splits["val"],
  norm=norm, device="cpu", cache=graph_cache)`로 검증 손실을 계산(기존 `graph_cache`를
  그대로 공유해 npz 재읽기 없음, W7-07 캐시 패턴 유지).
- 지금까지 중 검증 손실이 가장 낮은 에폭의 `state_dict`를 메모리에 들고 있다가, 루프가
  끝나면 **그 가중치를 `model.pt`로 저장**(마지막 에폭이 아니라).
- `metrics.json`에 에폭별 `loss`·`val_loss`·`epoch_time_s`를 모두 남기고, 선택된
  `best_epoch`/`best_val_loss`도 최상위에 기록 — 어떤 에폭이 선택됐는지 사후 확인 가능하게.
- `fine_tune_fn`의 시그니처·반환값(단일 `ckpt_path`)은 바꾸지 않는다 — `run_retrain_loop`
  (`retrain/loop.py`)는 이 함수를 `(champion_ckpt, cfg) -> ckpt_path`로만 호출하므로
  영향 없음. 인터페이스 동결 규칙(CLAUDE.md 6) 대상인 스키마·해시·종료코드는 변경 없음 —
  내부 알고리즘 변경이라 ADR은 쓰지만 "인터페이스 변경 승인"은 별도로 필요 없다고 판단.

## 비용·트레이드오프

- 매 에폭 `val` 10개 레시피 평가가 추가된다 — `train` 40~100개 평가보다 훨씬 작아(W7-07
  실측: fine-tune이 파이프라인의 92%를 차지하는 건 `train` 쪽이지, `val`은 그 15~25%
  규모) 상대적 오버헤드는 작다고 판단[제안, 실측 전 추정 — 실제 수치는 구현 후 재보고].
- 메모리에 베스트 `state_dict` 하나를 더 들고 있어야 함(MeshGraphNet 크기 기준 수십MB대
  — 문제 되는 규모 아님).

## 대안 (기각)

- **`ckpt_last.pt`/`ckpt_best.pt` 둘 다 디스크에 저장**(`train_main.py`와 완전히 동일한
  파일 구조): `fine_tune_fn`은 단일 `ckpt_path`를 반환해 `gate_fn`에 바로 넘기는 구조라,
  두 파일을 만들면 "게이트가 어느 파일을 보는지"를 추가로 배선해야 한다 — 베스트 하나만
  반환하는 지금 인터페이스를 유지하는 쪽이 변경 범위가 작다. 기각.
- **에폭마다 체크포인트를 디스크에 저장했다가 끝나고 베스트만 남기고 지움**: 디스크
  I/O가 늘고 `attempt*` 디렉터리에 중간 파일이 쌓인다(W9-01이 막 치운 것과 같은 문제를
  바로 다시 만듦) — 메모리에 best state_dict만 들고 있는 쪽이 낫다. 기각.
