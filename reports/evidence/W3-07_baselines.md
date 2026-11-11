# W3-07: 베이스라인 2종

- 날짜: 2026-11-08
- 코드: `src/etchpipe/models/baseline.py`(`DepthMlpExtrapolationModel` 신규, (a)는
  `models/dummy.py`의 `ZeroDisplacementModel` 재사용), `scripts/run_baselines.py`(평가 스크립트)
- 테스트: `tests/unit/test_baseline.py`(5건)

## 평가 방식 [확인]

두 베이스라인 모두 **최종 표면 1장 예측**(t=0 → t=actual_k) 기준으로 평가한다 —
체크리스트 D4 판정의 후퇴안("최종 프로파일 1장 예측")과 틀을 맞춰, W4-06에서 모델 A
롤아웃 오차와 바로 비교 가능하게 했다[제안: 이 선택의 근거].

- **(a) 변위 0**: 초기 표면(t=0)을 그대로 최종 예측으로 사용.
- **(b) 레시피→깊이 MLP 후 직선 외삽**: train split(300건)의 `meta.json["final_depth_um"]`로
  Θ(7차원, 정규화)→최종 깊이 회귀를 작은 MLP(torch, hidden=16, 2개 은닉층)로 학습한 뒤,
  val 레시피별로 초기 표면에서 `y<0`인 점(식각 영역)만 예측 깊이만큼 수직 이동 — 테이퍼·
  보잉 등 형상 변화는 모델링하지 않는 의도적으로 단순한 "직선(수직 이동)" 외삽. 예측 깊이가
  음수면 0으로 클램프(물리적으로 깊이는 음수가 될 수 없음).

지표: `chamfer_distance`, `depth_error`(둘 다 W2-06 기존 `eval/metrics.py` 재사용).

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_baseline.py -v
5 passed
  test_zero_displacement_returns_input_unchanged
  test_depth_mlp_fits_training_depths_reasonably   (fixture 3건 과적합, 오차<1.0µm)
  test_depth_mlp_predict_shifts_only_etched_points  (y>=0 점 불변, x 불변, y<0 점만 감소)
  test_depth_mlp_predict_depth_never_negative       (음수 타깃으로 학습해도 예측 깊이 클램프)
  test_depth_mlp_predict_raises_before_fit

PYTHONPATH=src python -m pytest tests/unit -q
81 passed (기존 76 + 신규 5)

python -m ruff check src/etchpipe/models/baseline.py tests/unit/test_baseline.py scripts/run_baselines.py
All checks passed!
```

## 실제 val(40건) 평가 결과 [확인, 구 Chamfer 정의 — 아래 "업데이트" 참고]

```
python scripts/run_baselines.py
```

| 베이스라인 | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) |
|---|---|---|---|---|
| (a) 변위 0 | 14.464 | 16.854 | 40.964 | 49.793 |
| (b) 깊이 MLP+직선 외삽 | 8.552 | 9.133 | 2.397 | 1.910 |

실행 시간 46초(train MLP 학습 포함, CPU). 레시피별 원본 수치:
`reports/evidence/W3-07_baseline_metrics.json`.

(b)가 (a)보다 깊이오차에서 약 17배, Chamfer에서 약 1.7배 낮다 — 레시피 Θ만으로도
최종 깊이의 상당 부분이 설명됨을 보여준다(기대한 방향). Chamfer가 깊이오차만큼 크게
줄지 않는 이유는 (b)가 테이퍼·보잉 등 형상 변화를 전혀 모델링하지 않고 수직 이동만
하기 때문 — 모델 A(MeshGraphNet)가 이 간극을 얼마나 줄이는지가 W4-06 비교표의 핵심
판단 대상이 될 것이다[제안: 해석].

## 영향 범위

- W4-06(롤아웃 오차 vs 베이스라인 비교표)에서 이 두 값을 그대로 비교 기준으로 쓸 수 있다.
- D4 판정("val 최종 스텝 Chamfer가 베이스라인 (b)보다 작으면 통과")의 기준값은 Chamfer
  평균 8.552(또는 중앙값 9.133) [확인: 실측값, 어느 쪽을 기준으로 할지는 W4에서 결정].

## 업데이트(2026-11-11): ADR-0004 Chamfer 정의 교체로 재계산 [확인, 1차 — 아래 2차로 대체]

`eval/metrics.py::chamfer_distance`가 점-점(cKDTree, 합)에서 점-선분(densify, 평균÷2)
방식으로 바뀌어(`docs/adr/0004-golden-scoring-chamfer-and-gate.md`), **위 Chamfer 수치는
전부 무효**. `depth_error`는 영향 없음(그대로). 같은 스크립트(`scripts/run_baselines.py`)
를 다시 돌려 `reports/evidence/W3-07_baseline_metrics.json`을 새 값으로 덮어썼다:

| 베이스라인 | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) |
|---|---|---|---|---|
| (a) 변위 0 | 7.700 | 9.318 | 40.964 | 49.793 |
| (b) 깊이 MLP+직선 외삽 | 0.590 | 0.582 | 2.397 | 1.910 |

## 업데이트(2026-11-11, 2차): outer_path 필터링 누락 수정 [확인 — 최종]

위 1차 재계산도 **원본 메시 점 배열을 그대로 `chamfer_distance`에 넘기는 버그**가
있었음을 발견 — `load_snapshot`의 점 배열은 마스크/Si 내부 경계가 섞여 순서가 안
맞는 부분이 있는데(실측: r0000 254개 선분 중 174개만 연속 인덱스), ADR-0004의
점-선분 Chamfer는 **표면을 따라 순서대로 연결된 곡선**을 가정한다. `scripts/
run_baselines.py`를 `etl.outer_path`로 표면선만 추출하도록 수정 후 재실행:

| 베이스라인 | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) |
|---|---|---|---|---|
| (a) 변위 0 | 8.143 | 9.620 | 40.964 | 49.793 |
| (b) 깊이 MLP+직선 외삽 | **0.438** | **0.351** | 2.397 | 1.910 |

**새 D4 판정 기준값은 Chamfer 평균 0.438(또는 중앙값 0.351)** — 위 "구 정의"
8.552/9.133과 "1차 재계산" 0.590/0.582는 역사적 기록으로 남겨두고 이 표로 대체한다.
`depth_error`는 `outer_path` 적용 여부와 무관(그대로, `depth()`가 순서에 안 의존).
