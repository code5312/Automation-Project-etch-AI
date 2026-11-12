# W6-08: `/predict` 실모델 연결

- 날짜: 2026-11-12
- 코드: `src/etchpipe/serve/inference.py`(신설), `src/etchpipe/serve/app.py`(재작성),
  `scripts/register_champion.py`(신설), `requirements-serve.txt`/`docker/Dockerfile.serve`
  (ViennaPS·physicsnemo 등 추가)

## 환경 변경 [확인, 사용자 결정 2026-11-12]

`requirements-serve.txt`가 원래 torch+fastapi만 가진 최소판(W1-03, 모델 아키텍처 확정
전)이었다 — 실제로 champion을 로드·추론하려면 훨씬 많은 의존성이 필요함을 이번에
발견했다. 최종 구성(`docker/Dockerfile.serve`):

```
torch==2.14.1, torchvision==0.29.1         # MeshGraphNetModel 전제
nvidia-physicsnemo==2.2.2, torch_geometric==2.8.1, torch_scatter==2.1.2
ViennaPS==4.5.0, ViennaLS==5.8.5           # t=0 초기 메시 구성(MakeTrench, 기하만)
pyvista==0.49.1, shapely==2.2.0, scipy==1.17.1, PyYAML==6.0.3  # etl.py/eval.py 체인
fastapi==0.162.0, uvicorn==0.54.0, pydantic==2.14.1, numpy==2.4.6
```

하나씩 실제로 빌드·import해보며 순서대로 발견(직접 import 테스트만으로는 못 잡음 —
`etl.py`/`eval.py`/`train.run_train`의 전체 import 체인까지 실행해봐야 드러남):
pyvista 누락 → shapely 누락 → scipy 누락 → PyYAML 누락 → torchvision 누락
(`physicsnemo.models` → `timm` → `torchvision`, 버전 안 맞으면
`RuntimeError: operator torchvision::nms does not exist`).

**"환경 3분리" 원칙에서 벗어남**: ViennaPS가 서빙 환경에도 들어갔다(t=0 메시의 점
밀도·분포가 ViennaLS 레벨셋 메싱 결과라 순수 Python 재현 불가, 실측 확인) — torch+
physicsnemo+ViennaPS를 같은 환경에 두고 import 테스트해 충돌 없음을 확인했다
(python:3.11-slim).

## champion 등록 [확인]

`scripts/register_champion.py`로 `runs/mgn_tuned/model.pt`를 레지스트리에 첫 champion
으로 등록(`registry/mgn_tuned_ep2/`) — W5-09(게이트 PASS)·`reports/eval/
mgn_tuned_val40_guarded`(지표)·실제 train split Θ 범위(`valid_range`)를 그대로 카드에
기록, 새로 계산한 값 없음.

## DOC-01: 범위 밖 입력 처리 [확인, 사용자 결정]

체크리스트 W6-08 원문("2배 초과 422")이 `project-definition.md` §4·`project-design.md`
§C7("유효 범위 밖은 거부하지 않고 `in_range=false`로 경고만 한다")과 어긋남 —
우선순위 규칙(definition > design > checklist)에 따라 상위 문서를 따름: **거부 안 함**.

## 결과 [확인] — 실제 champion으로 end-to-end 실행

```
GET /model (x-api-key 포함) -> 200
  model_version: mgn_tuned_ep2
  valid_range: {실제 train split Θ 범위 7개}
  gate_result: {passed: true, exit_code: 0, reasons: []}

POST /predict (recipe 1개, times=[2.0, 5.0]) -> 200
  model_version: mgn_tuned_ep2
  cd: 9.529, depth: 14.663, taper_deg: 3.673, in_range: true, clamped: false
  profiles: 2개(각 167개 표면선 점)
```

구성 요소: `ps.MakeTrench`(t=0 메시, 기하만) → `etl.to_graph` → `rollout`(fix_mask=True,
guard=True) → `outer_path`+`remove_self_crossing_loops`(표면선) → `mask_from_points`+
`cd_profile`+`taper_angle_deg`(cd/depth/taper).

## 알려진 한계 [미확인]

- **시간 해석**: 설계서 §7 예시는 분(min) 단위 연속값(0.25/0.5)을 쓰지만, 모델은
  `step_min`=1분 단위로만 학습됐다 — 요청 시간을 가장 가까운 정수 스텝으로 반올림한다
  (분 단위 보간은 검증된 적 없어 안 함, `clamped` 플래그로 `MAX_STEPS`(20) 초과 시
  표시).
- **champion 1개 고정 매핑**: `inference._ARTIFACT_PATHS`가 `model_id -> 체크포인트
  경로`를 하드코딩 — W6-04 루프가 실제로 새 champion을 승격하면 이 매핑도 늘려야 함.
- **rate limit**: 단일 프로세스 메모리 기준(분산 배포에는 부족, [제안]).

## 테스트

```
python -m ruff check src/etchpipe/serve/ scripts/register_champion.py tests/unit/test_serve.py
  → All checks passed
PYTHONPATH=src python -m pytest tests/unit/test_serve.py -q (etch-serve:0.1)
  → 19 passed
```

CI: `.github/workflows/pr-check.yml`에 `serve-tests` job 신설(requirements-serve.txt
전체 설치 후 `test_serve.py` 실행 — mock 기반이라 레지스트리/실제 체크포인트 없이도
통과).

## 후속 수정(2026-11-13): `pipeline-smoke` CI job 깨짐 — 실제 GitHub Actions에서 발견

브랜치를 처음 push한 뒤 실제 GitHub Actions에서 `pipeline-smoke` job이
`ModuleNotFoundError: No module named 'shapely'`로 실패함을 발견 — 재학습 루프와는
무관한 문제였다. 원인 두 가지:

1. **`inference.py`가 모듈 최상단에서 `torch`/`etl`/`eval`/`train.rollout`을
   import**하고 있어서, `pipeline-smoke`(더미 모델 전용, `requirements-dev.txt`만
   설치하는 가벼운 job)가 `etchpipe.serve.app`을 import만 해도 무거운 의존성이
   필요해졌다. → `inference.py`의 무거운 import를 전부 함수 안으로 지연시키고
   `from __future__ import annotations`로 타입 힌트도 미룸(viennaps는 이미 그렇게
   돼 있었음 — 나머지도 같은 패턴으로 맞춤).
2. **`shapely`가 `pyproject.toml`의 패키지 의존성에 원래부터 빠져 있었다**(W3-11에서
   `requirements-train.txt`에만 추가됨) — `run_pipeline_smoke.py` 자신의 1단계(ETL)가
   `etl.py`를 직접 쓰는데, `etl.py`가 내부 경계 필터링에 shapely가 꼭 필요해서
   `pip install -e .`만으로는 패키지가 실제로 안 돌아갔다. → `pyproject.toml`
   `dependencies`에 `shapely==2.2.0` 추가.
3. 위 두 가지를 고쳐도 `run_pipeline_smoke.py`의 6단계(`/predict` 실호출)는 실제
   champion·torch가 필요해 여전히 막힌다 — 이 스모크 테스트는 "더미 모델 전용"이
   원래 취지라, `tests/unit/test_serve.py`와 같은 패턴으로 `load_champion`/
   `predict_recipe`를 더미로 모킹하도록 수정.

검증: `python:3.11-slim` + `requirements-dev.txt` + `pip install -e .`만 설치한
(실제 `pipeline-smoke` job과 동일한) 컨테이너에서 `python scripts/run_pipeline_smoke.py`
실행 → `pipeline-smoke 통과` 확인. 실제 champion으로의 end-to-end `/predict`도
리팩터링 후 재확인(depth=14.66 — 이전과 동일값, 동작 변화 없음 확인).
`PYTHONPATH=src python -m pytest tests/unit -q`(fastapi 포함) → 201 passed.
