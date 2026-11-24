# W9-04: 재학습 루프 GPU(`device`) 지원 (ADR-0008)

## 1. 설계

ADR: `docs/adr/0008-retrain-loop-gpu-device-param.md`. 핵심: W8-09에서 확인된
"`fine_tune_fn`이 `device="cpu"`로 하드코딩돼 있어 자동 재학습 루프가 GPU를 못 쓴다"
(팀원 실측 92.9배, 품질 저하 없음)는 공백을 해소. 범위는 `fine_tune_fn`과 그 호출부
(`run()`, `run_retrain_demo.py`)로 한정 — 게이트 평가 경로는 제외.

## 2. 구현

- `src/etchpipe/retrain/execute.py::make_fine_tune_fn`에 `device: str = "cpu"` 선택
  인자 추가(기본값으로 기존 동작 100% 유지). 내부에서 `load_model_a(champion_ckpt,
  device=device)`·`train_one_epoch(..., device=device)`·`evaluate(..., device=device)`로
  하드코딩된 `"cpu"`를 교체.
- 체크포인트 저장 시 `best_state`/fallback `final_state` 모두 `.detach().to("cpu").clone()`
  — 어느 device로 학습했든 저장 파일은 항상 CPU 텐서(ADR-0008 결정 2, 레지스트리·서빙
  쪽 `map_location="cpu"` 가정과 호환).
- `run(device: str = "cpu") -> int`로 확장, `make_fine_tune_fn(..., device=device)`로
  전달.
- `src/etchpipe/retrain/__main__.py`: `run` 서브커맨드에 `--device {cpu,cuda}`
  (기본 `cpu`) 추가, `run(device=args.device)`로 연결.
- `scripts/run_retrain_demo.py`: 기존 `N_TRAIN`류 패턴과 동일하게 `DEVICE` 환경변수
  (기본 `"cpu"`) 추가, `make_fine_tune_fn(..., device=device)`로 전달, `result.json`
  요약에도 `"device"` 필드 기록. `.github/workflows/retrain-demo.yml`은 변경 없음
  (러너가 `ubuntu-latest`로 GPU 없음, 기존 CI 동작 그대로).

## 3. 테스트

기존 `tests/unit/test_execute_fine_tune.py`(W9-03) 2건의 몽키패치 더미 함수가 새
`device` 키워드를 못 받아 깨졌던 걸 수정(`fake_load_model_a(ckpt_path, device="cpu")`)
+ 신규 1건 추가:

- `test_fine_tune_fn_threads_device_to_load_train_evaluate` — `device="cuda"`로
  `make_fine_tune_fn`을 호출했을 때 `load_model_a`·`train_one_epoch`·`evaluate`
  전부가 실제로 `device="cuda"`를 받는지(`seen_devices` 리스트로 캡처) 확인 + 저장된
  체크포인트의 `state_dict` 텐서가 `device.type == "cpu"`인지 확인(ADR-0008 결정 2).
  실제 CUDA 없이도 몽키패치로 배선만 검증 — 로컬에 GPU 없음(CLAUDE.md "학습·추론
  CPU만").

```
$ PYTHONPATH=src python -m pytest tests/unit/test_execute_fine_tune.py -v
3 passed in 8.57s
$ python -m ruff check src/etchpipe/retrain/execute.py src/etchpipe/retrain/__main__.py \
    scripts/run_retrain_demo.py tests/unit/test_execute_fine_tune.py
All checks passed!
$ PYTHONPATH=src python -m etchpipe.retrain run --help
usage: python -m etchpipe.retrain run [-h] [--device {cpu,cuda}]
```

모든 기존 호출부(`scripts/run_retrain_demo.py`, `tests/unit/test_execute_fine_tune.py`)가
키워드 인자만 쓰고 있어 새 `device` 인자 추가로 인한 호출부 깨짐 없음(`grep -rn
"make_fine_tune_fn" src scripts tests` 전체 확인).

## 4. 미확인 / 다음 확인 필요

- **실제 GPU 호스트에서의 통합 실행은 미확인**[미확인] — 로컬에 GPU가 없고(CLAUDE.md
  환경 제약), Docker 데몬도 현재 미기동. W8-09의 92.9배는 팀원이 **별도 하네스**로
  재현한 수치이고, 이번에 하드코딩을 푼 코드 경로 자체로 같은 배율이 나오는지는
  GPU 환경이 되는 대로 `python -m etchpipe.retrain run --device cuda` 또는
  `DEVICE=cuda python scripts/run_retrain_demo.py`로 재확인 필요.
- 게이트 평가 경로(`make_gate_fn`)는 범위 제외 — 여전히 CPU 고정(ADR-0008 "범위"
  참고, 필요해지면 별도 ADR).
- `tests/unit/test_execute.py` 전체 재실행(회귀 확인)은 권한 분류기가 이번 세션에서
  차단해 돌리지 못함[미확인] — 다만 이 모듈은 로컬에 `DRIFT_RAW_DIR`이 없어 대부분
  skip되는 걸 이미 확인한 바 있어(W9-03 당시 조사) 핵심 회귀 리스크는 낮다고 판단.
