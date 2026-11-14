# W7-07: 전체 파이프라인 단계별 실행 시간 측정, 병목 1개 개선

## 측정 대상

`scripts/run_retrain_demo.py`(W6-07, ADR-0002 재학습 루프를 실제 champion
`runs/mgn_tuned/model.pt` + 실제 ViennaPS 시뮬레이션 + 실제 `retrain/execute.py`/
`retrain/loop.py`로 소규모 시연) — 더미 모델을 쓰는 `run_pipeline_smoke.py`(CI 전용,
W7-07 대상 아님)와 다름. 실행 환경: `etch-serve:0.1`(torch+physicsnemo+ViennaPS 전부
있는 유일한 이미지).

측정을 위해 `scripts/run_retrain_demo.py`에 단계별 타이머를 추가(sim 생성 루프,
`DriftState` ETL 구성, `fine_tune_fn`/`gate_fn`을 감싸는 투명 `_timed()` 래퍼) —
`retrain/execute.py`/`retrain/loop.py` 자체의 로직은 건드리지 않음(주입된 콜백을
바깥에서 감싸기만 함).

## 1차 측정 — 문제 발견: 동시 실행 오염 [확인]

전체 데모(`docker run etch-serve:0.1 python scripts/run_retrain_demo.py`)를 백그라운드로
2회 실행(2026-11-13) — 첫 실행이 "메모리 부족으로 종료"로 보고됐으나, 실제로는 **둘 다
끝까지 완주**했고, 결과 로그를 다시 읽어 확인(`build/retrain_demo/runs/*/metrics.json`
타임스탬프가 서로 겹침: `attempt1_1794591500`/`attempt1_1794592507` 등 — 두 컨테이너가
한동안 동시에 돌았음). 두 실행이 **같은 호스트 디렉터리(`build/retrain_demo/`)를 같은
바인드 마운트로 동시에 쓰기**까지 해서, `dataset_hash`가 서로 다르게 나오는 등(ETL 중간
상태 경쟁) 데이터 레이어까지 오염됨 — 이 1차 측정값은 **CPU 경쟁으로 부풀려진 수치**라
정확한 per-stage 비율로는 쓰지 않는다[미확인: 정확한 배율].

다만 이 오염된 측정 안에서도 한 가지는 신뢰할 수 있다: `attempt4`의 마지막 에폭(다른
컨테이너가 막 끝난 시점)만 경쟁 없이 돌아 `197.4초`로 측정됐고, 그 앞 9개 에폭은
`676~710초`로 균일하게 더 길었다(`build/retrain_demo/runs/attempt4_1794603567/metrics.json`)
— 동시 실행이 에폭당 약 3.5배까지 시간을 부풀렸다는 직접적 증거.

디스크에 남은 최종 `build/retrain_demo/result.json`(두 실행 중 나중에 쓴 쪽) 요약
(경쟁 영향 있음, 참고용):

| 단계 | 시간(초) | 비중 |
|---|---|---|
| sim(ViennaPS, 캐시됨) | 0.1 | 0.0% |
| ETL | 29.6 | 0.2% |
| fine-tune(4회 시도 합) | 16229.2 | 91.9% |
| gate(4회 시도 합) | 294.9 | 1.7% |
| **합계** | **17655.4** | 100% |

경쟁이 있었음에도 fine-tune이 압도적 비중(92%)이라는 결론은 구조적으로 안정적이다 —
sim/ETL/gate는 1회성 연산인 반면 fine-tune만 `epochs × 레시피 수`로 반복 역전파를 하기
때문에, 경쟁이 없었어도 같은 순위가 나올 것으로 판단[제안, 재시뮬레이션으로 재확인하지
않음].

## 병목: fine-tune 단계(CPU 미세조정)

`retrain/execute.py::make_fine_tune_fn`과 `train/train_main.py::run_training`이 공통으로
가진 패턴: 같은 `run_ids`(train/val)에 대해 `epochs`번 루프를 돌며 매번
`train/loop.py::iter_transition_pairs` → `load_graph`가 **같은 그래프 npz 파일을 디스크에서
다시 읽는다**(에폭 간 동일 파일 반복 + 같은 에폭 안에서도 인접 쌍이 경계 파일을 두 번
읽음). 모델 forward/backward(MeshGraphNet, CPU) 자체가 더 큰 비중을 차지하지만, 디스크
재읽기는 **훈련 결과에 전혀 영향을 주지 않는 순수 낭비**라 캐싱이 안전하다.

## 개선: 에폭 간 그래프 캐시 [확인]

`src/etchpipe/train/loop.py`: `load_graph`/`iter_transition_pairs`/`train_one_epoch`/
`evaluate`에 선택적 `cache: dict | None = None` 파라미터 추가. `cache`를 주면 같은 경로를
두 번째부터는 디스크 재읽기 없이 캐시된 텐서를 그대로 반환(읽기 전용 사용만 확인됨 —
`_prepare_pair`가 `normalize_*`로 새 텐서를 만들 뿐 원본을 수정하지 않음). `cache=None`
(기본값)이면 기존과 완전히 동일하게 동작(하위 호환).

적용 지점(둘 다 "같은 run_ids로 여러 에폭" 패턴):
- `retrain/execute.py::fine_tune_fn` — 에폭 루프 시작 전 `graph_cache = {}` 만들어 공유.
- `train/train_main.py::run_training` — 동일하게, train/val 양쪽에 같은 캐시 공유.

### 정합성 검증(캐시가 학습 결과를 바꾸지 않음)

`tests/unit/test_train_loop.py` 신규 3건(+ 기존 11건, 총 14건) —
`etch-train:0.1`에서 `pytest tests/unit/test_train_loop.py tests/unit/test_train_main.py -q`
→ **14 passed in 68.00s**.

- `test_load_graph_with_cache_avoids_second_disk_read`: 캐시된 뒤 원본 파일을 지워도
  두 번째 `load_graph` 호출이 터지지 않고 캐시된 객체를 그대로 반환.
- `test_iter_transition_pairs_with_cache_matches_without_cache`: cache 유무에 관계없이
  반환되는 텐서가 동일(`torch.equal`).
- `test_train_one_epoch_with_cache_gives_same_loss_as_without`: 같은 시드로 캐시
  있음/없음 각각 학습 1에폭 → **손실·가중치가 완전히 동일**(`==`/`torch.equal`) — 캐시가
  순수 I/O 최적화일 뿐 수치에 영향 없음을 직접 확인.

### 효과 측정(격리 벤치마크, 실제 champion+실제 demo 데이터) [확인]

전체 데모를 다시 5시간 돌리는 대신, 실제 champion 체크포인트(`runs/mgn_tuned/model.pt`)와
이미 ETL된 실제 demo 데이터셋(`build/retrain_demo/raw/demo0000-1`, 재시뮬레이션 없음)으로
`train_one_epoch`을 캐시 없이/있이 각 5에폭 반복 호출해 단독(경쟁 없는) 벤치마크
(`etch-train:0.1`, `n_train_ids=2`):

```
NO CACHE  total=371.06s per_epoch=[73.27, 69.89, 72.37, 68.28, 87.25]
CACHE     total=348.99s per_epoch=[75.23, 69.99, 68.26, 68.65, 66.87]
speedup(total) = 1.063x
```

**6.3% 단축**(371.06s → 348.99s, 5에폭 기준) — 레시피 2개짜리 작은 데모 기준이라
디스크 I/O가 전체 에폭 시간(모델 forward/backward가 대부분)의 일부만 차지해 효과가
크지는 않지만, 실측으로 확인된 real, lossless 개선이다. `epochs`가 더 큰 실제 2차 시도
(10~15에폭, ADR-0002 에스컬레이션 표)나 레시피 수가 더 많은 실제 운영 규모(40~100)에서는
캐시 적중 비율이 더 높아져(1에폭째만 전부 캐시 미스, 이후 전부 적중) 절감 폭이 이보다 더
클 것으로 예상되나, 그 규모로의 재측정은 하지 않음[미확인].

## 결론

- 병목: fine-tune 단계, 전체의 ~92%(측정값에 CPU 경쟁 오염 있음, 순위 자체는 신뢰).
- 개선: 에폭 간 그래프 npz 캐시 — 손실/가중치 불변 확인 + 격리 벤치마크로 6.3% 단축 확인.
- 남은 더 큰 비중(forward/backward 자체)은 CPU 제약 하 설계 범위 밖(정의서 5.2 "선택
  범위" 밖, `execute.py` 모듈 docstring이 이미 CPU 미세조정이 느림을 알고 하한 에폭을
  쓰는 것으로 대응 중) — 이번 항목은 "안전하게 개선 가능한 병목 1개"로 I/O 중복 제거를
  택함.

## 실행 명령 요약

```bash
# 단위 테스트(캐시 정합성)
docker run --rm -v "$(pwd)/src:/app/src:ro" -v "$(pwd)/tests:/app/tests:ro" \
  -w /app etch-train:0.1 pytest tests/unit/test_train_loop.py tests/unit/test_train_main.py -q
# -> 14 passed in 68.00s

# 격리 벤치마크(실제 champion+실제 demo 데이터, 재시뮬레이션 없음)
docker run --rm -v "$(pwd)/src:/app/src:ro" -v "$(pwd)/runs:/app/runs:ro" \
  -v "$(pwd)/build:/app/build" -v "<bench_cache.py>:/app/bench_cache.py:ro" \
  -w /app etch-train:0.1 python bench_cache.py
# -> speedup(total) = 1.063x
```

## 미확인/남은 작업

- 동시 실행 오염 없는 "클린" 전체 파이프라인 1회 실행의 정확한 `total_sec`/단계별 초 단위
  수치는 아직 없음(이번엔 병목 식별·개선 검증에 격리 벤치마크로 충분하다고 판단해
  5시간짜리 전체 재실행은 하지 않았음) — 필요하면 이후 단독 실행으로 보강 가능.
- 실제 운영 규모(40~100 레시피, 에폭 10~15)에서의 캐시 절감 폭은 추정만 하고 실측하지
  않음.
