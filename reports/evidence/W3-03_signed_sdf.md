# W3-03: ETL 완성 — 부호 있는 SDF, y 범위 자동 결정, 재료 매핑

- 날짜: 2026-11-03
- 코드: `src/etchpipe/etl/etl.py` (`signed_dist_field`, `_inside_solid`, `_dist_to_segments`, `auto_yr`)
- 테스트: `tests/unit/test_etl.py`
- 부호 규약·y범위 공식은 `docs/adr/0001-interface-freeze.md` §6이 ADR 작성을 요구한 항목이라
  `docs/adr/0003-signed-sdf-convention.md`로 동결(승인, 사용자 결정 2026-11-03).

## 1. 부호 있는 SDF (닫힌 폴리곤 가정 없이) [확인]

실제 스냅샷(`tests/fixtures/raw/*/snapshots/*.vtp`)의 경계선은 **닫힌 폴리곤이 아니라
좌/우 도메인 벽에서 끝나는 열린 체인**(`m.lines`가 점 수-1개, 양 끝 degree=1)이라는 걸
직접 확인했다 — 닫힌 폴리곤 point-in-polygon은 쓸 수 없음.

대신 **even-odd 규칙**(항상 solid인 바닥 `y_below`에서 쿼리점까지 수직 반직선을 쏴
경계 선분과 교차한 횟수의 홀짝으로 내부/외부 판정)을 선분 집합에 직접 적용했다.
이 방법은 체인 연결성(열린 체인, 중복점, 분기점)에 의존하지 않고 선분 교차 횟수만 세므로
언더컷·보잉(비단조 프로파일, 한 x에서 경계를 여러 번 지남)에도 그대로 맞는다.

부호 규약 [제안, 근거 없음 — 레벨셋/그래픽스 관례]: solid(마스크+Si) 내부 = 음수,
vacuum(식각된 공간) = 양수.

거리 값은 cKDTree(점 구름 최근접)가 아니라 **선분까지의 최단거리**로 계산(`_dist_to_segments`)
— 실측처럼 점이 촘촘하면 둘이 거의 같지만, 해석적 입력(점 2~6개짜리 직선·사각 트렌치)에서는
점 구름 최근접이 실제 거리와 크게 달라짐을 테스트 작성 중 직접 발견 → 선분 거리로 교체.

### 테스트 (해석적으로 답을 아는 입력)

```
PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q
14 passed
```

- `test_signed_dist_field_sign_and_value_straight_line`: 수평 직선, 아래=음수/위=양수, 값=수직거리(±0.3 이내)
- `test_signed_dist_field_rectangular_trench_sign_and_value`: 사각 트렌치 — 트렌치 안(부호+)/바닥 아래(부호-)/마스크 아래(부호-)/마스크 위(부호+), 바닥 경계 위 값≈0
- `test_signed_dist_field_bowing_void_inside_outside`: solid에 둘러싸인 식각 공극(보잉 단순화) — 한 x에서 경계를 2번 지나는 비단조 프로파일에서도 내부/외부가 올바름
- `test_signed_dist_field_no_nan`: NaN 없음

### 실제 스냅샷으로 사후 확인 [확인]

```
PYTHONPATH=src python -c "etl_run('tests/fixtures/raw/r0001', ..., THETA_KEYS)"
k=0 shape=(128,256) nan=False min=-53.00 max=14.58 solid비율=0.807
k=1 shape=(128,256) nan=False min=-52.99 max=14.58 solid비율=0.802
k=2 shape=(128,256) nan=False min=-52.25 max=14.58 solid비율=0.798
k=3 shape=(128,256) nan=False min=-51.48 max=14.58 solid비율=0.792
```

NaN 없음, 시간(k)이 지날수록 solid 비율이 줄어듦(식각 진행과 물리적으로 일치) — 정성적 타당성 확인.

## 2. y 범위 자동 결정 [확인]

기존 `dist_field`의 `yr=(-30, 12)`는 `configs/sim/base.yaml`의 `depth_max=50.0` 적용 **이전**
하드코딩 값이라, 지금 기준으로는 깊은 레시피의 바닥이 잘리는 버그였다(최대 -30까지만 담김,
실제로는 -50까지 가야 함).

`auto_yr(depth_max, mask_height_max, margin)`로 교체:
- `DEPTH_MAX = 50.0` — `configs/sim/base.yaml: sim.depth_max` [확인]
- `MASK_HEIGHT_MAX_OOD = 18.0` — `sampling.SPACE`의 `mask_height` hi=14.0에 `ood.extend=1.5`
  적용 시 최대값(`sampling.make_ood_recipes`와 동일 공식: `14 + (14-6)*(1.5-1) = 18.0`) [확인, 계산식 일치]
- `Y_MARGIN = 3.0` — 정지조건 오버슈트·그리드 경계효과 대비 여유 [제안]
- 결과: `yr = (-53.0, 21.0)` — `test_auto_yr_covers_depth_max_and_mask_height_with_margin`로 검증

x 범위(`xr=(-15,15)`)는 그대로 둠 — `run_sim.py`의 `ps.Domain(xExtent=30, ...)`가 `mask_width`와
무관하게 고정값이라 실제로 상수이지 자동 결정 대상이 아님 [확인, `run_sim.py:39`].

## 3. 재료 매핑 적용 [확인, 이미 완료]

`to_graph`의 `mat1h`(`mid<5`→mask, `mid>=5`→Si)가 `configs/etl/materials.yaml`(W1-S5 실측 확정,
0=mask/10=Si)과 일치함을 `test_to_graph_material_onehot_matches_w1_s5_mapping`이 이미
W2-04에서 검증하고 있었다 — 이번 작업에서 추가 변경 없음.

## 출력 형식 변경

`etl_run`이 `udf{k:03d}.npy`(부호 없음) 대신 `sdf{k:03d}.npy`(부호 있음)를 저장 — ADR-0001은
processed 파일명을 동결하지 않으므로 ADR 불필요 [확인, `docs/adr/0001-interface-freeze.md`에
`udf`/`sdf` 언급 없음]. 다운스트림(학습 코드)이 아직 없어 영향받는 소비자 없음.

## 남은 것

- rays=300/grid_delta=0.3 정확도 검증은 여전히 보류(`W3-01` 항목에 기록된 트리거 유지) — 이번
  signed SDF 사후 확인에서 눈에 띄는 표면 노이즈/이상은 없었음(재검토 트리거 미발생).
- `nx=256, ny=128` 격자 해상도는 기존 값 그대로 유지(변경 범위 아님).
