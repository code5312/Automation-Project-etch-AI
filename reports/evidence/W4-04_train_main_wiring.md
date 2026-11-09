# W4-04: 본 학습 진입점 배선([CPU/매핑])

- 날짜: 2026-11-10
- 코드: `src/etchpipe/train/train_main.py`(신규), `src/etchpipe/train/loop.py`(`train_one_epoch`에
  `device` 인자 추가)
- 테스트: `tests/unit/test_train_main.py`(신규 2건)

## 배경

W4-04(본 학습)는 지금까지 config(`configs/model/mgn_v1.yaml`)·모델(`MeshGraphNetModel`,
W4-03)·학습 루프(`train_one_epoch`, W4-02)가 각각 따로 있었고, 이 셋과 `init_run`(W4-01)을
엮어 "설정 파일 하나로 실제 학습을 돌리는" 진입점은 없었다. 이번에 그 배선만 작성했다 —
실제 장시간 실행(GPU)은 이 항목 범위 밖([GPU 팀원]).

## 구현

`run_training(model, optimizer, run_dir, graphs_root, train_ids, epochs, norm=None,
device="cpu", time_budget_s=None)`: 에폭마다 `train_one_epoch` 호출 → `metrics.json`(누적,
`{"epoch", "loss", "epoch_time_s"}`)과 `ckpt_last.pt`(`state_dict`+`epoch`)를 매번 덮어써서
중단돼도 마지막 에폭 결과가 남는다. `time_budget_s` 초과 시 그 에폭은 돌리지 않고
`{"stopped": "time_budget_exceeded", ...}`만 기록 후 멈춘다.

`build_model(config["model"])`: `MeshGraphNetModel`은 함수 안에서 지연 임포트 — physicsnemo가
없는 호스트에서도 `run_training` 자체는 토이 모델로 테스트 가능하게 분리.

`main(config_path, ...)`: config 로드 → `init_run`(W4-01, 시드·git_sha·packages.json 저장)
→ `data/processed/<dataset_hash>/{graphs,stats.json}`·`data/splits.json["train"]` 로드 →
`build_model`+`Adam(lr)` → `run_training` 호출. CLI(`--device`, `--time-budget-hours`,
`--out-root`, `--splits-path`) 포함.

`train_one_epoch`에 `device` 인자 추가: `norm`과 각 텐서를 호출마다 `.to(device)` —
`device="cpu"`(기본값)는 기존과 동일(무연산 `.to`)이라 기존 테스트·동작 변화 없음.

## 테스트

토이 모델(`nn.Linear`, physicsnemo 불필요) + 실제 fixture(`tests/fixtures/raw/r0000~r0002`,
`etl_run`으로 변환)로 배선만 검증:
- `test_run_training_writes_metrics_and_checkpoint_each_epoch`: 3에폭 실행 → `metrics.json`
  3건, `ckpt_last.pt`에 `epoch=2`/`state_dict` 존재.
- `test_run_training_stops_immediately_when_time_budget_already_exceeded`: `time_budget_s=-1`
  → 0에폭 실행, `ckpt_last.pt`/`metrics.json` 생성 안 됨.

```
docker run --rm --memory=2g -v "<repo>:/app" -w /app etch-train:0.1 bash -lc "
python -m ruff check src/ tests/   # All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
"
→ 95 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관), 그중 신규 2건 포함
```

## CPU 실행 가능성 실측 [확인] — 이 때문에 실제 실행은 GPU로 이관

`data/processed/4b8c0ce1bc98/graphs/`에서 train split(300개 run) 전이쌍 수를 직접 세어보면
**4,841쌍/에폭**. 2026-11-09 5-recipe 과적합 프로브(71쌍)가 에폭당 150~290초였으므로
쌍당 2.1~4.1초로 환산 → 전체 train 1에폭에 **약 2.8~5.5시간+** — 체크리스트 예산(3시간)을
1에폭도 못 채우고 넘긴다. 수렴에 필요한 에폭 수(보통 수십~수백)까지 고려하면 CPU로는
비현실적 — 이 결과 자체가 "GPU 팀원 이관" 결정(2026-11-09)이 맞았다는 재확인이다.

## GPU 실행 방법

```
PYTHONPATH=src python -m etchpipe.train.train_main configs/model/mgn_v1.yaml \
    --device cuda --time-budget-hours 3
```

- 입력 데이터: `data/processed/4b8c0ce1bc98/graphs/`, `data/processed/4b8c0ce1bc98/stats.json`
  (W3-04, 입력 정규화용), `data/splits.json`(`"train"` 키, 300개 run_id) — 모두 레포에 이미
  있음(데이터 자체는 `.gitignore`, 코드로 경로만 참조).
- 설정: `configs/model/mgn_v1.yaml`(`processor_size=6`, `hidden_dim_processor=64`,
  `lr=0.0001`, `epochs=100` — 마지막 둘은 [미확인: GPU 체감 속도 보고 재검토], 바꿀 경우
  사용자 확인 후 커밋).
- 출력: `runs/<run_id>/{config.yaml, git_sha, packages.json, metrics.json, ckpt_last.pt}`.
  `metrics.json`에 에폭별 `loss`/`epoch_time_s`가 쌓이므로 그대로 W4-04 "에폭당 시간 기록"
  증거로 쓸 수 있다.
- `--time-budget-hours`는 GPU 속도에 맞게 조정 가능 — 기존 "3시간" 예산은 CPU 기준
  [제안]이었으므로 GPU에서는 사용자 판단으로 늘리거나 줄여도 됨.
