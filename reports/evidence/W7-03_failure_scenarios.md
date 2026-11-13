# W7-03: 실패 시나리오 테스트

설계서 §3 C5 "자체 테스트" 확장 — 시뮬레이션 실패, 데이터 손상, 모델 로드 실패, 게이트
오탐/미탐 4개 범주를 각각 실제 코드로 검증했다.

## 1. 모델 로드 실패
`src/etchpipe/serve/inference.py::load_champion()`을 HTTP 레이어 모킹이 아니라
레지스트리 레이어에서 직접 검증(기존 `test_serve.py`는 `load_champion` 자체를
통째로 모킹해서 503 응답만 봤지, 레지스트리 조회·무결성 검증 로직은 격리 테스트한
적이 없었음). `tests/unit/test_inference.py`에 4개 추가:
- 레지스트리에 champion 미등록 → `ChampionUnavailable`
- 등록된 model_id의 체크포인트 경로를 못 찾음 → `ChampionUnavailable`
- 등록 후 아티팩트 바이트가 변조됨(SHA-256 불일치) → `ChampionUnavailable`
- 캐시 동작 확인(최초 1회만 `load_model_a` 호출)

이미 상당 부분 구현돼 있었음(`ChampionUnavailable` 예외 체계, 레지스트리
`verify_artifact` 변조 탐지) — 새로 발견한 버그는 없음, 격리 테스트만 보강.

## 2. 데이터 손상
`src/etchpipe/etl/etl.py::validate_run()`의 자체 docstring("예외를 던지지 않고
errors 리스트로 모아 반환 — 한 건이 전체 배치를 중단시키지 않도록")과 실제 동작이
달랐다. **버그 2건 발견·수정**:
1. `meta.json`이 없으면(시뮬레이션이 중간에 죽어 안 쓰인 경우) `FileNotFoundError`가
   그대로 전파 — `validate_dataset`로 여러 run을 배치 검증할 때 run 1개가 전체를
   중단시켰다.
2. 스냅샷 `.vtp`가 손상되면(쓰기 도중 끊김 등) `load_snapshot` 내부에서
   `KeyError: 'MaterialIds'`(VTK 파싱 실패 후)가 그대로 전파.

둘 다 try/except로 감싸 `errors` 리스트 항목으로 바꿨다. 회귀 테스트 3개 추가
(`tests/unit/test_etl.py`): meta.json 없음 단건, meta.json 없는 run이 섞인 배치가
나머지를 막지 않는지, 손상된 vtp 단건.

## 3. 시뮬레이션 실패
`src/etchpipe/sim/run_sim.py::run_sim()`은 예외를 전혀 안 잡는다 — 확인 결과 이는
**의도된 설계**다: 호출부(`sim_batch.py`, 사용자 지시로 내용 변경 안 함)가 이미
`try/except` + 1회 재시도 + 실패 기록(`failed_shard{N}.json`)으로 재시도 정책을
스스로 결정하는 구조라, `run_sim()`이 예외를 삼키면 오히려 그 재시도 로직이 깨진다.
그래서 `run_sim()` 자체는 고치지 않고, **"실패 시 디스크에 뭐가 남는지"**를 실제
ViennaPS로 확인했다(`tests/unit/test_run_sim.py`, `etch-sim:0.1` 컨테이너,
`ps.Process.apply`를 두 번째 호출에서 실패하도록 패치):
- 정상 경로: `meta.json` 작성, 스냅샷 개수 = `actual_k+1`.
- 실패 경로: `meta.json`은 **안 쓰임**(루프 끝에서만 씀), 실패 전까지의 부분
  스냅샷(`t000.vtp` 등)은 **남음**. 이 "메타 없고 부분 파일만 있는" 상태가 정확히
  위 2번(데이터 손상) 수정이 처리하는 상황 — 두 수정이 실제로 맞물린다.

## 4. 게이트 오탐/미탐
- **미탐**(나쁜 모델을 잘못 통과): W5-06 재검증(`reports/evidence/
  ADR-0005_W5-06_revalidation.md`)에서 노이즈/1에폭/시뮬레이터 변경 모델 3종을
  기존 경로·ADR-0005 신규 경로 양쪽으로 이미 검증 — 전부 정상 BLOCK.
- **오탐**(좋은 모델을 잘못 차단): 같은 재검증에서 **실제로 오탐 사례를 하나
  찾았다** — 정상 모델(mgn_tuned)이 golden-10(의도적 경계/극단 소표본)에서
  레시피 통과율 80%로 BLOCK. 원인은 게이트 로직 결함이 아니라 소표본·편향 표본에
  90% 기준을 적용한 결과(문서화된 한계, ADR-0005 참고). 이번에 추가로
  `tests/unit/test_gate.py::test_check_no_regression_does_not_false_block_a_
  better_candidate`를 보강 — 후보가 챔피언보다 일관되게 더 나을 때 윌콕슨
  단측검정(챔피언보다 나빠지는 방향만 봄)이 오탐으로 차단하지 않는지 확인(이전
  테스트는 "같음"/"나쁨" 방향만 있었고 "더 나음" 방향은 없었음).

## 검증
```
PYTHONPATH=src python -m pytest tests/unit/test_metrics.py tests/unit/test_gate.py \
  tests/unit/test_etl.py tests/unit/test_golden.py tests/unit/test_registry.py \
  tests/unit/test_inference.py -q -k "not cache"
-> 129 passed, 1 deselected

docker run ... etch-train:0.1 pytest tests/unit -q --ignore=test_serve.py --ignore=test_run_sim.py
-> 215 passed (전체 스위트, 회귀 없음)

docker run ... etch-sim:0.1 (pytest 임시 설치) pytest tests/unit/test_run_sim.py -v
-> 2 passed (실제 ViennaPS 프로세스로 검증)
```

## 미확인 / 남은 작업
- "정상 모델 80% BLOCK" 오탐이 실제 승격 판정 규모(golden_drift10+val10=20개,
  val 40개)에서도 재현되는지는 여전히 미확인(ADR-0005 작업 때부터 남은 항목).
- `sim_batch.py`(재시도+실패 기록 로직)는 사용자 지시로 내용을 건드리지 않기로 한
  스파이크 코드라 직접 테스트를 새로 추가하지 않았다 — 코드 읽고 설계 의도만 확인.
