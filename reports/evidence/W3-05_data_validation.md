# W3-05: 데이터 검증 스크립트

- 날짜: 2026-11-07
- 코드: `src/etchpipe/etl/etl.py`(`validate_run`, `validate_dataset` 신규),
  `scripts/validate_data.py`(CLI 래퍼)
- 테스트: `tests/unit/test_etl.py`
  - `test_validate_run_passes_on_fixture`, `test_validate_dataset_aggregates_fixture_runs`
  - `test_validate_run_detects_snapshot_count_mismatch`
  - `test_validate_run_detects_nan_points`
  - `test_validate_run_detects_non_monotonic_depth`

## 검증 항목 [확인]

`validate_run(run_dir)`가 run 1건에 대해 확인하는 것(체크리스트 W3-05 그대로):
1. 스냅샷 개수 == `meta.json`의 `actual_k+1`(조기종료 시 K보다 작을 수 있음을 반영)
2. 점 수>0
3. 점·법선에 NaN/Inf 없음(`np.isfinite`)
4. 식각 깊이가 시간에 따라 감소하지 않음(물리 타당성, 설계서 §9)
5. 점 수 최대치(그래프 크기 예산, 설계서 §4.2 C2 위험 항목) — 전체 run 중 최댓값으로 집계

예외를 던지지 않고 `errors` 리스트로 모아 반환 — 배치 검증 중 한 건이 전체를 중단시키지
않도록 설계.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q
25 passed   (기존 20건 + 신규 5건)

PYTHONPATH=src python -m pytest tests/unit -q
69 passed   (전체 스위트 회귀 없음)

python -m ruff check src/etchpipe/etl/etl.py tests/unit/test_etl.py scripts/validate_data.py
All checks passed!
```

## 실제 N=400 데이터 검증 실행 [확인]

```
PYTHONPATH=src python scripts/validate_data.py data/raw
```

```
splits.json 사용: 400건 (스파이크 흔적 등 그 외 디렉터리는 제외)
400 runs checked, 0 with errors
max_n_points (그래프 크기 예산) = 663
```

실행 시간 2분 43초.

### `data/raw`를 디렉터리 전체 스캔한 1차 시도에서 발견한 문제 [확인]

`scripts/validate_data.py`가 처음엔 `data/raw` 아래 `meta.json`이 있는 디렉터리를
전부 검사하도록 했는데, `KeyError: 'actual_k'`로 실패했다. 원인: `data/raw`에
1~2주차 스파이크 산출물 13건(`r0000_repeat`, `r0000_s1k2`~`r0007_s1k2`, `s3_r0`~`s3_r3`,
이전 백업 작업에서 이미 "본 데이터셋 N=400과 무관한 하위호환 leftover"로 확인된
것들)이 섞여 있는데, 이 옛 스키마의 `meta.json`에는 `actual_k` 필드가 없다(당시엔
`K` 고정값만 썼음). **수정**: 스크립트가 `data/splits.json`(본 데이터셋의 공식
run_id 400개 목록)이 있으면 그것만 검사하도록 바꿔(`data/raw` 디렉터리 스캔은
`splits.json`이 없을 때의 대체 경로로만 유지), 본 데이터셋과 무관한 흔적을 자동으로
제외시켰다.

## 영향 범위

- 그래프 크기 예산(점 수 최대치) **663**은 모델 A(MeshGraphNet) 쪽 배치·메모리 설계에
  참고할 상한값으로 쓸 수 있다 [확인: 실측값, 추가 해석 없음].
- N=400 전체(train300+val40+test40+ood20, W3-11·W3-12 버그 수정분 포함)에 물리적
  이상(비NaN, 점수>0, 깊이 비감소, 스냅샷 개수 일치) 없음을 확인 — 베이스라인(W3-07)·
  골든셋(W3-06) 작업을 신뢰하고 진행할 수 있는 근거.
