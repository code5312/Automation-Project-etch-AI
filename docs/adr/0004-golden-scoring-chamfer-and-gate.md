# ADR-0004: Chamfer 정의 통일 + 회귀 게이트 규칙 재구성 (골든셋 공식 채점 노트북 기준)

- 날짜: 2026-11-11
- 상태: **승인(사용자 결정 2026-11-11)**
- CLAUDE.md 작업규칙 6: "인터페이스는 2주차 말에 동결. 변경은 ADR 작성 후에만." — 다만
  설계서 §3 C5의 규칙 ①~⑤는 애초에 `[제안, 베이스라인 후 보정]`으로 명시되어 "나중에
  실측 기준으로 고친다"고 예고된 항목이다. 이 ADR이 그 보정이다.
- 영향 문서: `docs/project-design.md` §3 C4/C5, `src/etchpipe/eval/metrics.py`,
  `src/etchpipe/gate/gate.py`, `tests/unit/test_metrics.py`, `tests/unit/test_gate.py`,
  `docs/project-checklist.md` (W5-01/02, W5-05)

## 배경

팀원이 전달한 `golden10_채점_(입력용).ipynb`(+ `README_FNO담당.md`/`README_MGN담당.md`)을
확인했다. 이 노트북은:

- 팀 저장소의 **고정 커밋**(`8575ff5f758f07886cf4f73a379a0da868adaaa8`)에서 골든셋을
  받아 `params_hash`로 무결성까지 검증한다. 우리 `golden/golden.json`의 `golden_hash`
  (`e95bdca4de55`)와 정확히 일치 — 같은 골든셋 [확인, 교차검증].
- MGN·FNO 양쪽 담당자 README가 완전히 같은 기준·코드를 참조 — **모델 간 공통, 실제
  합/불 판정에 쓰이는 채점 기준**으로 판단된다.
- 노트북 자체가 "여기 Chamfer는 팀 `gate.py`의 Chamfer와 계산 방식이 달라 값을 직접
  비교할 수 없다"고 명시 — 우리 쪽이 재정의 대상임을 노트북이 스스로 지적.

## 결정 1: Chamfer 정의를 노트북 방식으로 교체 [확인, 노트북 코드 그대로 포팅]

**기존** (`eval/metrics.py::chamfer_distance`, cKDTree):

```
d_pred_to_true = 점→점 최근접(cKDTree)
chamfer = d_pred_to_true.mean() + d_true_to_pred.mean()   # 합, ÷2 없음
```

**신규** (노트북 `chamfer()`/`densify()`/`distance_to_curve()` 그대로):

```
pred', true' = densify(pred, 0.05µm), densify(true, 0.05µm)   # 선분 위에 점 보충
d_pred_to_true = 점→선분 최단거리(각 점에서 상대 곡선의 선분에 수직 투영, 구간 clip)
chamfer = 0.5 * (d_pred_to_true.mean() + d_true_to_pred.mean())   # 평균
```

차이 2가지: **①점-점 vs 점-선분**(점 밀도가 다르면 점-점 방식이 왜곡됨 — ADR-0003 §4가
SDF 거리 계산에서 이미 같은 이유로 선분 방식을 택한 전례와 일치), **②합 vs 평균**(기존
값은 신규 값의 약 2배).

**영향 범위**:
- `eval/metrics.py::chamfer_distance` 교체 — `eval.py`(W5-01/02 골든셋 리포트), `gate.py`
  rule①, `train/rollout.py`(평가용 롤아웃 오차)가 전부 새 값을 쓰게 됨.
- `train/chamfer_loss.py`(학습 손실)는 **그대로 유지** — 미분 가능해야 해서 `torch.cdist`
  점-점 방식을 쓰는데, 속도·역전파 때문에 의도적으로 다른 구현이다. 다만 그 파일 docstring이
  "`eval/metrics.py`와 같은 정의"라고 적어 놓은 부분은 더 이상 사실이 아니므로 같이 고친다.
- **기존에 측정된 모든 Chamfer 수치는 무효** — 새 정의로 재계산 필요. `reports/evidence/`의
  과거 수치(W4-06 롤아웃 비교표 등)는 "구 정의 기준"이라고 표시해 혼동을 막는다(재계산은
  범위 밖, 다음 학습 라운드에서 자연히 새 값으로 대체).

## 결정 2: 회귀 게이트 — 차단 기준을 노트북 방식으로 재구성 [확인, 노트북 9번 셀 그대로]

노트북이 실제로 쓰는 합격 조건(①③ 제외, ②④+물리만 차단):

| 설계서 규칙 | 노트북에서 | 이 ADR 이후 |
|---|---|---|
| ① 골든 평균 Chamfer ≤ 임계 | **차단에서 제외**, 참고 지표만 | 참고 지표로 다운그레이드(차단 안 함) |
| ② CD·깊이 오차 ≤5% | 차단 — 단 정답<6µm면 **절대오차 ≤0.3µm**로 분기 | 분기 로직 그대로 포팅 |
| ③ 직전 모델 대비 악화 ≤10% | **차단에서 제외** | 참고 지표로 다운그레이드(차단 안 함) |
| ④ 롤아웃 후반/전반 ≤2.0 | 차단 | 유지(집계 축은 결정 3) |
| ⑤ 추론 시간 상한 | 경고만(차단 안 함) | 변경 없음 |
| (신설) 물리 | 선 꼬임 수=0 **+** 모델별 결함 수(수동 입력)=0 | 신설 — 결정 4 |

①③을 차단에서 빼는 이유(노트북 작성자 의도, 우리도 동의): ①은 골든셋 자체가 train
레시피라 "일반화"가 아니라 "학습 회차 비교"용 숫자라 절대 상한을 못 박기 애매하고, ③은
`check_champion_regression`이 이미 "챔피언 없음 → 자동 통과"를 다루지만 "직전보다 나빠도
CD/깊이/롤아웃이 다 기준 내"인 모델을 막을 이유가 약하다는 판단.

`GateResult`에 차단에는 안 쓰지만 보고는 해야 하는 ①③ 수치를 남길 필드가 필요 —
기존 `warnings`(⑤ 전용, 차단 안 하지만 "문제" 신호)와 섞지 않고 `reference_metrics: dict`
(중립적 "참고용" 숫자, 경고가 아님)를 신설한다.

## 결정 3: ④ 집계 축을 "여러 레시피 스텝별 평균"으로 변경 [확인, 노트북 `late_early_ratio` 그대로]

**기존** `check_rollout_late_step`: 레시피 1개의 `step_errors` 리스트를 그대로 반/반.

**신규**: 골든셋 10개 레시피의 Chamfer를 **스텝 인덱스 기준으로 먼저 평균**한 시퀀스
(`chamfer_by_step[k] = mean(chamfer at step k across recipes that reach step k)`)를
반/반 나눠 비율을 본다. 레시피 하나만으로는 식각 공정 특유의 스텝별 변동(레시피마다 깊이
도달 속도가 다름)이 노이즈로 보일 수 있어, 여러 레시피를 묶어야 "후반에 갈수록 정말
나빠지는지"를 더 안정적으로 보여준다는 노트북 설계 의도에 동의.

제로 가드도 포팅: `errors.max() < 1e-6` 이면(오차가 사실상 0) 비율 대신 `1.0` 반환 —
계산 잔차로 비율이 터지는 것 방지. 기존 `check_rollout_late_step(step_errors: list[float])`
시그니처는 **레시피 1개용으로 이름을 남기고**, 새 함수
`check_rollout_late_step_multi(step_errors_by_recipe: list[list[float]])`를 추가해
`run_gate`가 둘 다 받을 수 있게 한다(레시피 1개짜리 호출부와도 하위 호환).

## 결정 4: "물리" 규칙 신설 — 선 꼬임 자동 계산 + 모델별 결함 수동 입력 [확인/미확인 혼재]

코드베이스 전체를 확인했고(`src/etchpipe/`), **self-intersection(선 꼬임)을 세는 함수가
현재 없다** [확인 — `design.md`에도 언급은 있으나(§3 C5 "자체 테스트") 구현은 없음].
신설 필요:

- `self_intersection_count(curve)`(`eval/metrics.py`): 노트북 `self_intersections()`
  포팅(인접하지 않은 선분끼리 교차 + 격자 1칸보다 작은 고리는 무시). 게이트는 이 값을
  직접 계산하지 않고 `int` 결과만 받는다 — 순수 지표 함수는 C5(게이트)보다 C4(평가)에
  두는 게 레이어링상 맞다[배선 완료 2026-11-11, `reports/evidence/ADR-0004_chamfer_gate_rework.md`
  "배선" 절].
- 모델별 결함 수(MGN=마스크 영역 점 이동, FNO=떨어진 조각)는 노트북도 **자동 계산하지
  않고 사람이 직접 센 값을 입력**한다(`MODEL_CHECK_VIOLATIONS`) — 우리 게이트도 같은
  구조로, `run_gate`에 `model_specific_violations: int | None = None` 파라미터를 추가해
  `None`이면 "미확인"으로 통과시키되 결과에 표시하고, 값이 있으면 `>0`일 때 차단한다
  [제안, 노트북의 "미확인 표시" 동작을 차단 로직으로 옮긴 것 — 노트북은 `None`이어도
  통과시키지만 우리 게이트는 자동화 파이프라인이라 "미확인 상태로 PASS시켜 배포"하면
  위험하므로 더 보수적으로 간다. 사용자 확인 필요].

## 대안 (기각)

- 기존 `chamfer_distance`를 유지하고 노트북 수치만 별도로 병행 기록: 두 값이 계속
  따로 돌면 "진짜 기준이 뭐냐"는 혼동이 반복될 것이므로 기각, 하나로 통일.
- ①③을 그대로 차단 기준에 유지하고 노트북은 "참고용 별도 채점"으로 완전히 분리: 그러면
  우리 게이트가 노트북보다 더 엄격해져서 "팀 공식 채점은 합격인데 우리 파이프라인은
  차단"하는 모순이 생길 수 있어 기각, 노트북 기준에 맞춤.

## 미확인 / 다음 확인 필요

- 결정 4의 `model_specific_violations=None` 처리 방식(미확인으로 보수적 차단 vs 노트북처럼
  통과+표시)은 사용자 확인 필요.
- 노트북·README 원본 파일(`golden10_채점_(입력용).ipynb`, `README_FNO담당.md`,
  `README_MGN담당.md`)은 **커밋하지 않고 작업 참고용으로만 둔다**[사용자 결정
  2026-11-11]. `.gitignore`에 추가해 `git status`에 안 잡히게 했다.
