# W5-01: 평가 지표 전체 구현([CPU/매핑])

- 날짜: 2026-11-10
- 코드: `src/etchpipe/eval/metrics.py`(IoU, CD_top/mid/bot, taper, bowing, latency 추가),
  `src/etchpipe/train/rollout.py`(`rollout_val_error_by_step` 추가)
- 테스트: `tests/unit/test_metrics.py`(신규 16건), `tests/unit/test_rollout.py`(신규 2건)

## 배경

지금까지(W2-06) `chamfer_distance`/`depth`/`depth_error`만 있었고, 설계서 §3 C4가 정의한
나머지 지표(IoU, CD_top/mid/bot, taper, bowing, 추론 시간)와 "스텝 5·10·20 보고"는
W5-01로 남아 있었다. 체크포인트 생성(W4-04/06)이 GPU 팀원(현재 Hugging Face 플랫폼에서
MGN/FNO 학습·롤아웃 진행 중, 2026-11-10 결정)에게 넘어간 뒤에도, 지표 구현 자체는
체크포인트 의존성이 없어 바로 진행했다.

## 구현

### 공정 비교(설계서 §3 C4): 모델 A/B를 같은 마스크로

모델 A(점 집합+연결)와 모델 B(SDF, W6-11 이후)의 출력을 같은 기준으로 비교하기 위해
둘 다 **래스터 마스크**(solid=True, `etl.signed_dist_field`와 같은 부호 규약)로 변환한다.

- `mask_from_points(p, lines, nx, ny, xr, yr)`: `etl.signed_dist_field`를 거쳐가므로
  내부 경계 필터링(마스크/Si 접촉면 제거, ADR-0003)을 그대로 상속 — 새 geometry 로직을
  만들지 않고 이미 테스트된 SDF 파이프라인을 재사용.
- `mask_from_sdf(sdf)`: `sdf <= 0`. 모델 B(FNO)가 나오면 이 한 줄로 바로 연결 가능(아직
  FNO 미구현, W6-11에서 실제로 맞는지 재확인 필요 [미확인]).

### 지표

- `iou(mask_pred, mask_true)`: 식각된(vacuum) 영역 기준 IoU [제안, 변하는 건 트렌치
  모양이라 vacuum으로 계산].
- `cd_at_y(mask, xr, yr, y_target)`: 지정한 깊이(y)에서 중심(x=0)을 포함해 바깥으로
  뻗어나가는 연속 vacuum 구간의 폭.
- `cd_profile(mask, xr, yr, depth_value, bottom_frac=0.9)`: top(y=0)/mid/bot. bot은
  바닥 끝점(폭이 0으로 수렴하는 불안정 지점)이 아니라 깊이의 90% 지점에서 측정
  [제안, 끝점 불안정성 회피].
- `taper_angle_deg(cd_top, cd_bot, depth_value)`: top→bot 선형 프로파일 가정
  `tan(theta) = (cd_top - cd_bot) / 2 / depth` [제안, 정의서 7.2 "테이퍼 각도"의 구체 구현].
- `bowing(cd_top, cd_mid, cd_bot)`: `cd_mid - (cd_top+cd_bot)/2` (양수=바깥으로 볼록)
  [제안].
- `measure_latency(fn, *args, **kwargs)`: `time.perf_counter()` 기반 실행 시간 측정.

### 스텝 5·10·20 보고

- `rollout_val_error_by_step(model, graphs_root, run_id, steps, step_min, norm=None)`:
  `max(steps)`까지 한 번만 롤아웃하고, `steps`(예: `[5,10,20]`)의 각 스텝에서 실제
  `g<step>.npz`와 비교. 지금까지(W4-05/06)는 최종 스텝만 봐서 중간에 오차가 급증했다가
  다시 줄어드는 패턴을 놓칠 수 있었다(`reports/model_issues_mgn.md` "판단 기준" 3번,
  C5 규칙④ "롤아웃 후반 오차 급증 없음"도 중간 스텝 없이는 제대로 못 봄).
- `rollout_val_error`는 이제 `rollout_val_error_by_step(..., [n_steps])[n_steps]`을
  부르는 얇은 래퍼로 재구성 — 기존 호출부(`scripts/run_rollout_comparison.py`,
  `tests/unit/test_rollout.py`)는 동작 그대로.

## 테스트

해석적으로 답을 아는 합성 데이터로 검증(CLAUDE.md 규칙 3·4):
- `mask_from_points`: `test_etl.py`의 사각 트렌치 점·선분(마스크 top y=5, 측벽 x=±2,
  바닥 y=-3)을 그대로 재사용해 마스크 변환 결과 확인.
- IoU: 동일 마스크(1.0), 분리된 두 vacuum 영역(0.0), 폭 4 겹침 폭 2(1/3) — 직접 계산한
  합성 사각/사다리꼴 트렌치 마스크로 검증.
- CD: 수직 측벽 사각 트렌치는 top=mid=bot, 중심이 solid면 0.
- taper/bowing: 선형 사다리꼴(top>mid>bot), 수직 측벽이면 각도 0, `atan` 계산값과
  직접 대조.
- `rollout_val_error_by_step`: 실제 fixture(`r0000`, 스텝 1~3)로 스텝별 결과 확인 +
  기존 `rollout_val_error`와 같은 스텝에서 결과 일치 확인(래퍼 리팩터링 검증).

```
docker run --rm --memory=2g -v "<repo>:/app" -w /app etch-train:0.1 bash -lc "
python -m ruff check src/ tests/   # All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
"
→ 113 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관), 신규 18건 포함
```

## 다음 (미착수)

- W5-02(`eval.py`): 위 지표들을 체크포인트+`dataset_hash`로 묶어 `metrics.json`/PNG로
  출력하는 진입점 — 아직 없음.
- W5-04/05(게이트 임계값·완성): `gate.py`는 아직 규칙①④만(W2-07). 오늘 추가한 CD/taper/
  bowing/latency는 게이트 규칙②③⑤에 바로 연결되지는 않음 — 임계값 산출(W5-04)부터 필요.
- 실제 체크포인트(팀원이 Hugging Face에서 학습 중)로 이 지표들을 돌려본 적 없음
  [미확인] — 지금까지는 합성 데이터로만 함수 정확성을 검증했다.
