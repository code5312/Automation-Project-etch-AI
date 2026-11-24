# ADR-0008: 자동 재학습 루프에 GPU(`device`) 파라미터 추가

- 날짜: 2026-11-24
- 상태: **승인(사용자 결정 2026-11-24, W9-04 진행 지시)**
- CLAUDE.md 작업규칙 6: 인터페이스(스키마·해시 규칙·종료 코드)는 2주차 말 동결, 변경은
  ADR 작성 후에만. `make_fine_tune_fn`/`run()`에 새 선택 인자를 추가하는 것으로, 기존
  호출(인자 생략)은 전부 `device="cpu"`로 전과 동일하게 동작 — 스키마·해시·종료 코드
  자체는 바뀌지 않지만 CLAUDE.md 환경 절("학습·추론 CPU만")을 사실상 확장하는
  결정이라 ADR로 남긴다.
- 영향 파일: `src/etchpipe/retrain/execute.py::make_fine_tune_fn`/`run`,
  `src/etchpipe/retrain/__main__.py`, `scripts/run_retrain_demo.py`,
  `docs/project-checklist.md` W9-04.
- **범위 제외**: `make_gate_fn`(게이트 평가 경로)은 CPU 고정 유지 — 아래 "범위" 참고.
  `.github/workflows/retrain-demo.yml`(CI, `ubuntu-latest`에는 GPU가 없어 적용 대상
  아님) 변경 없음.

## 배경

W8-09에서 팀원이 외부 GPU(RTX A5000)로 같은 설정(3에폭, Adam lr=1e-4)을 별도 하네스로
재현해, CPU 대비 **92.9배** 빠르면서 품질은 떨어지지 않음을 실측했다
(`reports/evidence/W8-09_gpu_feasibility.md`). 그런데 그 실행은 저장소의
`python -m etchpipe.retrain run`을 그대로 쓴 게 아니었다 — `execute.py::make_fine_tune_fn`의
`fine_tune_fn`이 `train_one_epoch(..., device="cpu", ...)`로 **하드코딩**돼 있어서
(`device` 인자 자체가 없음) 자동 재학습 루프 자체는 GPU를 쓸 수 없었다. W7-07에서
fine-tune 단계가 전체 파이프라인의 92% 병목이라고 실측된 바 있어, 이 하드코딩을 푸는
효과가 크다고 판단한다.

## 결정

**범위**: `fine_tune_fn`과 그 호출부(`run()`, `run_retrain_demo.py`)만 — 게이트 평가
경로(`make_gate_fn`, `_eval_recipes`, `rollout_val_error_by_step`)는 이 ADR에 포함하지
않는다. 이유: (1) 게이트는 매 시도 1~2회만 돌아 fine-tune처럼 반복 비용이 쌓이지 않고,
(2) `rollout_val_error_by_step`까지 device를 꿰려면 추가 조사가 필요해 "검증된 적
없는 코드를 사실로 쓰지 않는다"(CLAUDE.md 4항) 원칙상 범위를 넓히지 않는다 — 필요해지면
별도 ADR.

1. `make_fine_tune_fn(state, fallback_norm, out_root=RETRAIN_RUNS_ROOT, device="cpu")`에
   `device` 선택 인자 추가(기본값 `"cpu"` — 생략 시 기존과 100% 동일). 내부적으로
   `load_model_a(champion_ckpt, device=device)`·`train_one_epoch(..., device=device)`·
   `evaluate(..., device=device)`로 그대로 전달한다 — 이 세 함수는 W4-04부터 이미
   `device` 인자를 받도록 설계돼 있었다(`train_main.py`가 이미 이 경로로 GPU를 쓴다,
   W8-09 §1) — `execute.py`만 하드코딩돼 있던 것이었다.
2. **체크포인트는 항상 CPU 텐서로 저장**한다 — `device="cuda"`로 학습했어도 저장 직전
   `state_dict`를 `.cpu()`로 옮긴다. `load_model_a`가 항상 `map_location="cpu"`로 읽은
   뒤 `.to(device)`하므로 꼭 필요한 변경은 아니지만, 레지스트리·서빙 쪽의 기존 가정
   ("체크포인트 파일은 CPU 텐서")을 명시적으로 지켜 어떤 호스트에서 학습했는지와 무관하게
   읽는 쪽 코드가 전혀 안 바뀌게 한다.
3. `run(device: str = "cpu")`에 같은 선택 인자 추가, `make_fine_tune_fn(..., device=device)`로
   전달. CLI(`python -m etchpipe.retrain run --device {cpu,cuda}`, 기본 `cpu`)로 노출
   — `etchpipe/retrain/__main__.py`의 `argparse`에 `--device` 추가.
4. `scripts/run_retrain_demo.py`는 기존 `N_TRAIN`류 환경변수 패턴과 동일하게 `DEVICE`
   환경변수(기본 `"cpu"`)로 받아 `make_fine_tune_fn(..., device=device)`에 전달. CI
   워크플로우(`retrain-demo.yml`)는 바꾸지 않는다 — `ubuntu-latest` 러너에 GPU가 없어
   `DEVICE`를 지정해도 쓸 수 없고, 지정하지 않으면 기존과 동일하게 CPU로 돈다.

## 비용·트레이드오프

- 실제 GPU 호스트에서의 통합 실행은 이번 작업에서 재확인하지 못한다[미확인] — 로컬에
  GPU가 없고(CLAUDE.md "학습·추론 CPU만"), Docker 데몬도 현재 미기동. W8-09의 92.9배는
  **별도 하네스**의 실측이라 이 코드 변경 이후 실제로 같은 배율이 나오는지는 다음에
  GPU 환경이 되는 대로 `python -m etchpipe.retrain run --device cuda`로 재확인해야
  한다 — 이번 항목은 "하드코딩을 풀어 가능하게 만든다"까지이고 "실측 재현"은 포함하지
  않는다.
- `load_model_a`가 매 시도 `champion_ckpt`를 다시 읽을 때도 `device=device`로 로드돼
  GPU 메모리에 모델을 올리는 비용이 매 시도 반복된다 — ADR-0002 에스컬레이션(최대
  4회 시도) 기준 무시할 수 있는 규모로 판단[제안].

## 대안 (기각)

- **환경변수(`ETCH_RETRAIN_DEVICE`)로만 노출, 함수 시그니처는 그대로**: `run()`/CLI를
  건드리지 않아 변경 범위는 더 작지만, `make_fine_tune_fn`을 직접 호출하는
  `run_retrain_demo.py`에서는 결국 같은 선택 인자가 필요해 환경변수 하나로는 두 호출부
  일관성을 못 맞춘다 — 기각.
- **게이트 평가까지 한 번에 GPU화**: 범위가 커지고 `rollout_val_error_by_step` 쪽
  검증이 안 된 상태라 기각(위 "범위" 설명 참고) — 필요해지면 별도 ADR.
