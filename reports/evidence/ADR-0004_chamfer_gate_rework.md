# ADR-0004 적용: Chamfer 정의 교체 + 게이트 규칙 재구성 ([CPU/매핑])

- 날짜: 2026-11-11
- 결정 문서: `docs/adr/0004-golden-scoring-chamfer-and-gate.md`
- 코드: `src/etchpipe/eval/metrics.py`, `src/etchpipe/train/chamfer_loss.py`(docstring만),
  `src/etchpipe/gate/gate.py`, `scripts/run_gate_negative.py`, `scripts/run_pipeline_smoke.py`
  (주석만)
- 테스트: `tests/unit/test_metrics.py`, `tests/unit/test_chamfer_loss.py`,
  `tests/unit/test_gate.py`(전면 재작성)

## 배경

팀원이 전달한 골든셋 공식 채점 노트북(`golden10_채점_(입력용).ipynb` + MGN/FNO 담당
README)을 확인한 결과, 우리 `chamfer_distance`·게이트 규칙과 정의가 달랐다(상세 근거는
ADR-0004 본문). 이 노트북이 팀 공통으로 쓰는 실제 합/불 판정 기준으로 판단해 그대로
채택했다.

## 결정 1: Chamfer 정의 교체

`eval/metrics.py::chamfer_distance`를 점-점(cKDTree) + 합(÷2 없음)에서 **점-선분
(densify 0.05µm + 선분 투영) + 평균(÷2)**으로 교체. `_densify_curve`/`_distance_to_curve`
신설(노트북 코드 그대로 포팅). `scipy.spatial.cKDTree` 의존 제거(이제 numpy만).

`train/chamfer_loss.py`(학습 손실)는 **구현은 그대로**(역전파·속도 때문에 점-점 유지)
— docstring만 "더 이상 같은 정의 아님"으로 수정. `test_chamfer_loss.py`의 교차검증
테스트도 `eval.chamfer_distance` 대신 독립 참조 구현(`_reference_point_point_chamfer`,
cKDTree)과 비교하도록 수정.

## 결정 2+3: 게이트 규칙 재구성

- `GateResult`에 `reference_metrics: dict` 필드 신설.
- ①(골든 Chamfer)·③(챔피언 대비 악화)은 계산은 하지만 더 이상 `passed`를 안 가름 —
  `reference_metrics["golden_chamfer"]`/`["champion_regression"]`에만 기록.
- `check_cd_depth_error`(②) 시그니처를 사전 계산된 %오차 리스트 대신 **원본 깊이/CD
  값**(`depth_pred/true`, `cd_pred/true`)+`grid`를 받도록 변경, 내부에서 신설
  `check_dimension_error`(노트북 `dimension_rule()` 포팅 — 정답<6µm면 절대오차 분기)로
  깊이·CD 각각 판정 후 AND.
- `check_rollout_late_step_multi`(④, 노트북 `late_early_ratio()` 포팅) 신설 — 여러
  레시피의 스텝별 오차를 스텝 인덱스 기준 평균 후 반/반 비율, 제로 가드 포함. 기존
  `check_rollout_late_step`(레시피 1개용)은 호환용으로 유지, `run_gate`가 다중 레시피
  인자(`rollout_step_errors_by_recipe`)를 우선한다.
- `check_self_intersections`(물리, 노트북 `self_intersections()` 포팅) 신설, `run_gate`에
  `self_intersection_count`/`model_specific_violations` 추가 — 둘 다 **아직 안 넘기면
  (None) 그 부분만 건너뛴다**[사용자 결정 2026-11-11, 기존 ②③⑤ 관례와 동일하게 보수적
  차단 대신 skip].
- `passed = ok2 and ok4 and ok_physics`로 변경(기존 `ok1 and ok2 and ok3 and ok4`).

`scripts/run_gate_negative.py`의 "무작위 변위 모델" 케이스는 ①이 더 이상 안 막아서
exit_code=1을 더 이상 보장 못 함 — 깊이/CD 오차를 직접 주입하는 케이스로 교체.

## 테스트

```
docker run --rm --memory=2g -v "<repo>:/app" -w /app etch-train:0.1 bash -lc "
python -m ruff check src/ scripts/ tests/   # All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
"
→ 152 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관 — 매 세션 동일)
```

신규/수정 테스트: `test_metrics.py`에 희소 점 케이스 추가(점-점이면 d≈5.59로 과대평가,
점-선분이면 d=2.5로 정확 — 교체 이유를 직접 검증), `test_gate.py` 전면 재작성(①③
참고지표화, 다중 레시피 ④, `check_dimension_error` 분기, 물리 규칙 신설·skip 동작).

## 영향/주의

- **기존에 측정된 모든 Chamfer 수치는 무효**(새 정의로 재계산 필요) — `reports/evidence/`
  과거 수치는 "구 정의 기준"으로 읽어야 한다. 재계산은 범위 밖(다음 학습 라운드에서 자연
  대체).
- W5-01/W5-02/W5-05는 체크박스를 유지한다(기능 자체는 완료) — 다만 Chamfer·게이트 판정
  기준이 ADR-0004로 바뀌었으니 두 항목 평가·게이트를 다시 돌릴 땐 이 문서를 같이 본다.

## 배선 (2026-11-11 후속, 같은 날)

물리 규칙의 자동 계산 부분을 실제 평가 파이프라인에 연결:

- `check_self_intersections`를 `eval/metrics.py::self_intersection_count`로 재배치
  (게이트는 이 값을 직접 계산하지 않고 호출부가 미리 구해서 넘기는 구조라, 순수 지표
  함수는 C5가 아니라 C4에 두는 게 레이어링상 맞음 — `gate.py`는 더 가벼워짐).
- `etl.py`에 `outer_path(points, lines)` 신설(노트북 포팅) — 마스크/Si 내부 경계가
  섞인 전체 메시 연결에서 플라즈마에 닿는 표면선만 추출.
- `train/rollout.py`: `lines_from_edge_index`(`to_graph`의 양방향 엣지 역산) +
  `rollout_val_error_by_step(..., check_physics=True)` — 스텝마다 `self_intersections`
  계산. 예측이 도메인 밖으로 발산해 `outer_path`가 끝점을 못 찾으면(실측: 정규화 없이
  토이 모델에 실제 θ 크기를 그대로 먹인 경우 1스텝에 발산) 예외를 전파하지 않고
  `self_intersections=1`(명백한 물리 위반)로 집계.
- `eval/eval.py::run_eval`: `check_physics=True`로 호출해 `by_step`/`final_step`에
  `self_intersections_total`을 추가. **`run_gate` 호출은 안 함** — eval.py의 책임은
  "체크포인트→metrics.json"까지이고(C4), 게이트 호출은 C5 별도 단계(W5-08)로 남긴다.
- 테스트: `test_etl.py`(outer_path, 가지 제외), `test_metrics.py`(self_intersection_count),
  `test_rollout.py`(물리 체크 켬/끔, 발산 폴백), `test_eval.py`(출력 구조) 갱신.

```
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
→ 157 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관)
```

## 다음 (미착수)

- `model_specific_violations`(MGN 마스크 이동, FNO 떨어진 조각 수)는 노트북처럼 여전히
  사람이 세어 넣는 수동 입력 — 자동화 대상 아님.
- CD/깊이/taper/bowing의 정확한 측정 높이·기준(정답 깊이의 10/50/90% 등)을 노트북과
  맞추는 작업은 범위 밖(이번엔 Chamfer·게이트 판정 기준만) — 다음 확인 필요.
- `eval.py` 출력(`self_intersections_total`)을 실제로 `run_gate(self_intersection_count=...)`
  에 넘기는 호출부(W5-08, CI `gate-negative`를 실모델 기준으로 교체)는 여전히 범위 밖.
- 참고: 팀원이 `runs/mgn_tuned/`를 `runs/fno_tuned/`와 같은 구조(README/config/metrics/
  골든10 채점 결과 CSV)로 갱신함(2026-11-11) — 다음 턴에서 MGN 실측 수치 분석 필요
  [미확인, 아직 안 읽음].
