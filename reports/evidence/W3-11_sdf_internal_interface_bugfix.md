# W3-11 (계획 외 추가, 사용자 제보로 발견): SDF 내부 재료 경계 버그 수정

- 날짜: 2026-11-06
- 배경: 사용자가 "ViennaPS를 SDF로 변환 시 마스크와 Si 사이 경계까지 빈 공간과의
  경계로 계산해서 일부 값이 잘못 나온다"는 제보를 전달. 코드 투영·실측으로 확인한
  결과 실제 버그였음(아래).
- 코드: `src/etchpipe/etl/etl.py` (`_filter_internal_interfaces` 신규, `signed_dist_field`
  가 이를 적용하도록 수정)
- 테스트: `tests/unit/test_etl.py::test_signed_dist_field_excludes_internal_material_interface`(신규)
- 새 의존성: `shapely==2.2.0` (`requirements-train.txt`, 사용자 확인 후 도입)

## 버그 원인 [확인]

ViennaPS가 `saveSurfaceMesh(..., addInterfaces=True)`로 내보내는 메시(`run_sim.py`가
이미 이 옵션으로 저장 중)에는 solid-vacuum 외곽 경계뿐 아니라, 마스크가 덮어서 아직
vacuum에 노출되지 않은 Si 상단(solid-solid 내부 경계)도 같은 선분 배열에 섞여 있다.

`tests/fixtures/raw/r0001/snapshots/t000.vtp`로 실측:
- 전체 218개 선분 중 58개(마스크 왼쪽 밑 29개 + 오른쪽 밑 29개, 둘 다 도메인 좌우
  경계에서 끝나는 open chain)가 내부 경계였다.
- **MaterialIds만으로는 내부/외곽을 구분할 수 없음을 직접 확인** — 내부 경계와,
  트렌치 오프닝에서 실제로 vacuum과 맞닿는 외곽 경계가 둘 다 `MaterialIds=10`으로
  동일했다.
- 내부 경계를 그대로 두면 `_inside_solid`(짝수-홀수 교차 규칙)가 깨져서, 마스크
  몸통 안쪽(예: x=-10, y=0.001~1.0, 실제로는 솔리드)이 `inside_solid=False`로,
  마스크 위 공간(예: x=-10, y=20, 실제로는 vacuum)이 `inside_solid=True`로
  **둘 다 반대로** 나왔다.
- (처음에 x=0 근처 트렌치 오프닝 중심을 "마스크 내부"로 잘못 가정하고 조사했다가,
  실제 MakeTrench 지오메트리는 마스크가 도메인 좌우 바깥쪽을 덮고 중앙에 트렌치
  오프닝이 있는 구조임을 다시 확인해 재조사·정정함.)

## 수정 방법 [확인]

선분 soup 전체(외곽+내부 경계+열린 체인을 닫기 위한 임시 보조선)를 `shapely`의
`polygonize`+`unary_union`으로 닫힌 다각형들로 쪼갠 뒤 union하면, 두 다각형이
공유하는 내부 경계는 union 결과의 둘레(boundary)에서 사라지고 내부로 들어간다 —
그 둘레 위에 남아있는 원래 선분만 골라서 쓴다(새 함수 `_filter_internal_interfaces`).
임시 보조선(열린 체인을 닫기 위해 바닥 y_below까지 내린 선분들)은 반환하지 않음 —
거리 계산(`_dist_to_segments`)을 오염시키지 않기 위함.

구현 중 실제로 걸린 문제: 열린 체인의 양 끝(도메인 좌우 경계)이 Si쪽·마스크쪽 두
지점으로 따로 존재해서(점 4개), 각각을 바닥으로 내리는 보조선이 서로 겹치는
(collinear overlap) 구간을 만들어 `polygonize`가 다각형을 하나도 못 찾았다(실측
확인) — `unary_union`으로 선분들을 먼저 노드 처리한 뒤 `polygonize`를 적용해
해결함.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q
16 passed   (기존 15건 전부 그대로 통과 + 신규 회귀 테스트 1건)

PYTHONPATH=src python -m pytest tests/unit -q
59 passed   (전체 스위트 회귀 없음)

PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/run_pipeline_smoke.py
pipeline-smoke 통과 (dataset_hash c44329191626, 기존과 동일 — SDF 내용은
dataset_hash 계산에 들어가지 않으므로 변경 없음이 맞음)
```

실제 fixture(r0001 t000.vtp)로 수정 전/후 비교:
- (x=-10, y=0.001/0.5/1.0, 마스크 몸통 안쪽): 수정 전 `False`(오류) → 수정 후 `True`(정상)
- (x=-10, y=20, 마스크 위 vacuum): 수정 전 `True`(오류) → 수정 후 `False`(정상)
- (x=0, y=0.5/20, 트렌치 오프닝): 수정 전/후 모두 `False` — 이 지점은 원래부터 맞았음

## 영향 범위

- `signed_dist_field`(모델 B/FNO용 SDF)만 영향. `to_graph`(모델 A/MeshGraphNet용
  그래프 특징)는 재료 one-hot을 선분 soup 전체에서 점 단위로 평균 내는 방식이라
  이 버그와 무관.
- **실제 운영 데이터(N=400)에 영향 없음** — 확인 시점에 `data/processed/`가 비어
  있어(= 실 데이터에 `process_dataset`을 아직 한 번도 돌리지 않음) SDF가 잘못
  계산된 production 산출물이 존재하지 않았다. raw(.vtp)/GitHub Release/Hugging
  Face에 올린 백업도 raw 데이터이므로 이 버그와 무관.
- 기존 CI 픽스처(`tests/fixtures/raw`)로 만든 `sdf*.npy`(테스트 산출물)는 재생성
  대상이지만 테스트 자체가 `tmp_path`에 매번 새로 만들므로 별도 정리 불필요.

## 남은 것

- W3-04(정규화 통계 `stats.json`) 착수 시 이 수정이 반영된 코드로 실제 N=400
  데이터에 처음 `process_dataset`을 돌리게 됨 — 순서상 안전함.
