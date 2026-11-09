# W4-05: 롤아웃 v1(연결 유지, 법선·엣지 재계산, 재료 이전 값 유지)

- 날짜: 2026-11-09
- 코드: `src/etchpipe/train/rollout.py`(신규) — `recompute_edge_features`, `recompute_normals`,
  `rollout`, `rollout_val_error`. `src/etchpipe/train/loop.py`의 `_load_graph`를
  `load_graph`로 공개 전환(롤아웃에서 재사용).
- 테스트: `tests/unit/test_rollout.py`(6건, 신규)

## [CPU/매핑] 범위 — 이번에 한 것

자기회귀 롤아웃(모델 자신의 예측을 다음 입력으로 다시 먹임, 팀포싱 W4-02와 다름) 메커니즘을
구현하고 **미학습 토이 모델**로 배관을 검증했다:

- **연결 유지**: `edge_index`는 입력 그대로 매 스텝 동일하게 모델에 전달 — 점 추가·삭제
  없음을 가정.
- **법선·엣지 재계산**: 엣지는 `etl.to_graph`와 같은 정의(rel_x, rel_y, 거리)로 새 `pos`에서
  재계산. 법선은 세그먼트 수직 벡터를 꼭짓점별 평균하는데, 수직 방향이 두 가지(±90도)라
  **이전 스텝 법선과 내적이 양수인 쪽**으로 부호를 고정(연속성 가정 — ETL이 VTK `Normals`로
  구하는 "진짜" 바깥쪽 방향을 다시 풀 수는 없음, 회전이 매 스텝 작다고 가정).
- **재료 유지**: material one-hot은 입력 그래프 값을 그대로 들고 가고 재계산하지 않음.
- **t 증가**: `etl.etl_run`과 같은 정의(`k * step_min`)로 스텝마다 `step_min`만큼 더함.

## 테스트 [확인]

손으로 계산 가능한 간단한 기하(수평선 3점)로 `recompute_edge_features`/`recompute_normals`
단위 공식을 직접 검증하고, 두 가지 토이 모델로 `rollout` 전체 배관을 검증했다:

- `RecordingZeroModel`(변위 항상 0): 모델에 들어가는 `node`/`edge_index`를 스텝마다 기록해
  연결(`edge_index`)·재료(`node[:,4:6]`)·theta(`node[:,6:13]`)가 안 바뀌고 `t`(`node[:,13]`)만
  `i*step_min`씩 느는지 직접 대조.
- `MaterialEchoModel`(변위 = 자기 입력의 material 채널): 재료가 매 스텝 제대로 전달되는지
  위치 누적값(`pos0 + n_steps*material0`)과 비교 — 재료가 중간에 깨지면 이 등식이 깨짐.
- `rollout_val_error`: 실제 ViennaPS fixture(`tests/fixtures/raw/r0000`, actual_k=3)를
  `etl_run`으로 그래프 변환해 토이 `nn.Linear` 모델로 끝까지 실행, `eval/metrics.py`의
  `chamfer_distance`/`depth_error`로 마지막 스텝 오차 계산까지 플러밍 확인(수치 자체는
  토이 모델이라 의미 없음, W4-04 본 학습 체크포인트로 교체 시 그대로 재사용).

```
PYTHONPATH=src python -m pytest tests/unit/test_rollout.py -v → 6 passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
  → 93 passed, 3 failed (test_run_train.py git_sha 관련 — etch-train:0.1 이미지에 git
    미설치, 오늘 변경과 무관한 기존 환경 한계)
python -m ruff check src/etchpipe/train/ tests/unit/test_rollout.py → All checks passed
```

## 범위에서 제외한 것 — **[GPU 팀원]**

- **실제 val 오차 기록**: 지금은 토이 모델로 플러밍만 확인했다. 실제 "K스텝 예측, val 오차
  기록"은 W4-04 본 학습(GPU 팀원, `processor_size=12`) 체크포인트가 나온 뒤
  `rollout_val_error(model, graphs_root, run_id, n_steps, step_min, norm=load_norm_stats(...))`를
  val 40개 run에 돌려서 채운다.
- **법선 부호 연속성 가정의 검증**: 회전이 매 스텝 작다는 가정이 실제 데이터(특히 보잉·
  언더컷처럼 곡률이 큰 구간)에서도 유지되는지는 실제 모델 롤아웃 결과로만 확인 가능 —
  [미확인].
