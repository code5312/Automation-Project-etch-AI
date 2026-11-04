# W3-09: ETL 성능 — process_dataset 레시피 단위 병렬화

- 날짜: 2026-11-03
- 배경: 코드 리뷰(`/code-review`가 아니라 직접 구조 분석)에서 `signed_dist_field`가
  선분 루프를 Python `for`로 돌며 격자 32,768점 전체에 벡터 연산을 적용함을 실측 —
  실제 fixture(231 선분)에서 1회 0.13~0.15초, `process_dataset`(3레시피×4스냅샷)
  2.03초. N=400 본 데이터 전체로는 수 분~20분대 순차 CPU 시간으로 추정.
- 코드: `src/etchpipe/etl/etl.py` (`process_dataset`에 `max_workers` 파라미터 추가)
- 테스트: `tests/unit/test_etl.py::test_process_dataset_parallel_matches_serial`

## 먼저 틀린 가설을 실측으로 걸러냄 [확인]

"선분 루프를 (E,N) 브로드캐스트로 완전 벡터화하면 빠를 것"이라는 가설을 실제 구현해
벤치마크했으나 **6배 느려짐**(0.148초 → 0.928초, 같은 fixture) — E×N≈760만 원소 임시
배열을 여러 개(`t`,`px`,`py`,`dist`,`y_at`,`in_x`,`cross`) 만들면서 메모리 대역폭이
병목이 됨. 이 방향은 채택하지 않음.

## 적용한 개선: run_id 단위 ProcessPoolExecutor [확인]

`process_dataset`은 run_id마다 출력 경로가 독립적이라(공유 상태 없음)
embarrassingly-parallel 케이스. `max_workers` 파라미터를 추가해 호출자가 명시적으로
켤 수 있게 함.

```
PYTHONPATH=src python bench_parallel.py   # tests/fixtures/raw 3레시피, __main__ 가드 포함
serial sec: 3.952      dataset_hash: c44329191626
pool-2 sec: 4.945       dataset_hash: c44329191626
pool-auto sec: 4.881    dataset_hash: c44329191626
```

**레시피 3개 규모에서는 풀 생성 자체의 고정 비용(Windows 실측 약 1초/워커)이 이득보다
커서 오히려 느려짐** — 그래서 기본값은 `max_workers=1`(기존 동작과 완전히 동일, 회귀
없음)로 두고, N=400 규모의 본 데이터 생성 같은 대량 호출에서만 호출자가
`max_workers`를 명시적으로 올리도록 설계했다. 호스트 CPU 코어 14개 확인
(`multiprocessing.cpu_count()`).

세 모드(serial/pool-2/pool-auto) 모두 `dataset_hash`가 완전히 동일 — 병렬화가 결과를
바꾸지 않음을 확인.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q
15 passed
```

`test_process_dataset_parallel_matches_serial`: `max_workers=1`(직렬)과 `max_workers=2`
(풀)로 각각 돌려 `manifest` 전체(딕셔너리 동등), 각 run의 `metrics.json` 바이트,
각 `sdf{k:03d}.npy` 배열(`np.array_equal`)이 모두 동일함을 확인.

## 남은 것

- 실제 N=400 본 데이터 생성(W3-02, 현재 백그라운드 진행 중)에 이 병렬화를 적용하려면
  `scripts/gen_splits.py`나 W3-04 쪽 호출부에서 `max_workers`를 명시적으로 넘기는
  코드가 추가로 필요함 — 이번 항목은 `process_dataset` 자체의 병렬화 옵션 추가까지만
  다룸.
- 실측은 Windows 호스트 기준. kind(Linux 컨테이너) 안에서 실행되는 ETL 단계가 생기면
  fork 방식이라 프로세스 생성 비용이 더 낮을 가능성 — 재측정 필요 시 트리거로 기록.
