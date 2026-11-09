# ADR-0003: 부호 있는 SDF 좌표계·부호 규칙 동결

- 날짜: 2026-11-03
- 상태: **승인(사용자 결정 2026-11-03)** (ADR-0001 §6이 "W3-03에서 확정 후 ADR-0002 예정"이라고
  적어 뒀지만, ADR-0002 번호는 그 사이 재학습 자동 반복 결정(2026-11-02)에 먼저 쓰였다. 이
  문서가 그 자리를 대신하는 ADR-0003이다.)
- CLAUDE.md 작업규칙 6: "인터페이스는 2주차 말에 동결. 변경은 ADR 작성 후에만." — SDF는
  ADR-0001 §6에서 "아직 동결하지 않은 것"으로 명시적으로 남겨둔 항목이라, 지금 처음 정하는
  것이지 동결된 걸 바꾸는 게 아니다.
- 영향 문서: `docs/adr/0001-interface-freeze.md` §6, `docs/project-design.md` C2,
  `src/etchpipe/etl/etl.py`

## 결정

### 1. 부호 규약 [제안, 근거 없음 — 다른 곳에 참조할 기존 값이 없어 레벨셋/그래픽스 관례를 따름]

- solid(마스크+Si) 내부 = **음수**
- vacuum(식각된 공간) = **양수**

### 2. y 범위 자동 결정 공식 [확인, `configs/sim/base.yaml`·`sampling.SPACE` 값에서 계산]

```
auto_yr(depth_max, mask_height_max, margin) = (-(depth_max + margin), mask_height_max + margin)
DEPTH_MAX = 50.0              # configs/sim/base.yaml: sim.depth_max
MASK_HEIGHT_MAX_OOD = 18.0     # sampling.SPACE mask_height hi=14.0 + ood.extend=1.5 적용 최대값
Y_MARGIN = 3.0                 # [제안] 정지조건 오버슈트·그리드 경계효과 대비 여유
```

결과: `yr = (-53.0, 21.0)`. `base.yaml`의 `depth_max`나 `sampling.SPACE`의 `mask_height` 범위가
바뀌면 이 상수들도 같이 바꿔야 한다(자동이 아니라 "config 값 기반으로 계산"이라는 뜻).

x 범위는 자동 결정 대상이 아니다 — `run_sim.py`의 `ps.Domain(xExtent=30, ...)`가 `mask_width`와
무관한 고정값이라 `xr=(-15,15)`는 상수로 유지 [확인, `run_sim.py:39`].

### 3. 내부/외부 판정 방법 [확인, 테스트로 검증]

실제 ViennaPS 스냅샷의 경계선은 닫힌 폴리곤이 아니라 도메인 벽에서 끝나는 열린 체인이라
point-in-polygon을 못 쓴다 — **even-odd 규칙**(항상 solid인 바닥에서 쿼리점까지 수직
반직선을 쏴 경계 선분과 교차한 횟수의 홀짝)으로 판정한다. 체인 연결성에 의존하지 않아
언더컷·보잉(비단조 프로파일)에도 맞는다.

### 4. 거리 계산 방법 [확인, 해석적 테스트로 발견·수정]

점 구름 최근접(`cKDTree(p).query`)이 아니라 **선분까지의 최단거리**로 계산한다. 실측처럼
점이 촘촘하면 둘이 비슷하지만, 점이 희소한 영역에서는 점 구름 방식이 실제 거리와 크게
달라짐을 테스트 작성 중 발견했다.

### 5. 출력 파일명

`data/processed/<dataset_hash>/graphs/<run_id>/sdf{k:03d}.npy` (부호 없는 `udf{k:03d}.npy`를
대체). `manifest.json`/`splits.json`(ADR-0001 §2) 스키마는 변경 없음 — 그래프 폴더 안의
파일명만 바뀜.

## 근거

- `src/etchpipe/etl/etl.py` (`signed_dist_field`, `_inside_solid`, `_dist_to_segments`, `auto_yr`)
- `tests/unit/test_etl.py` — 직선·사각 트렌치 부호/값, 보잉(언더컷) 내부/외부, NaN 없음 (14 passed)
- `reports/evidence/W3-03_signed_sdf.md` — 실제 fixture로 사후 확인(NaN 없음, 시간에 따라 solid 비율 감소)

## 대안 (기각)

- 점 구름 최근접 거리 유지: 해석적 입력에서 값이 크게 틀려서 기각.
- 체인을 닫힌 폴리곤으로 억지로 이어붙여 point-in-polygon 적용: 분기점·중복점 토폴로지가
  실측 데이터마다 달라 체인 연결 순서에 의존하는 방법은 깨지기 쉽다고 판단해 기각.

## 업데이트 (2026-11-09): y 범위 공식 변경 [확인, 사용자 결정 2026-11-09]

GPU 가용 팀원이 FNO(모델 B) 초안(`runs/fno_draft/best.mdlus`, train 300·val 40 기준 실측
val Chamfer 중앙값 0.118µm — 이미 실전급 결과 보유)을 이 ADR의 원래 공식(바닥 margin만
적용, `yr=(-53,21)`)과 다른 범위(`yr=(-68,21)`, 즉 바닥에 `mask_height_max`만큼 추가)로
학습해 전달함. 팀원이 이미 완료한 학습 결과를 재활용하기 위해 **바닥 공식을 변경**:

```
auto_yr(depth_max, mask_height_max, margin) = (-(depth_max + mask_height_max), mask_height_max + margin)
```

결과: `yr = (-68.0, 21.0)`(위 범위는 변경 없음). §2의 원래 공식·결과값은 역사적 기록으로
남겨두고 이 섹션으로 대체한다.

**영향**: `data/processed/<dataset_hash>/sdf/`(이전 범위로 생성된 거리장)는 재생성이
필요하다 — graphs/는 `auto_yr`을 쓰지 않아 영향 없음, `dataset_hash`도 영향 없음(simulation
파라미터 기반이라 그리드 정의와 무관). 재생성 일정은 [미확인].
