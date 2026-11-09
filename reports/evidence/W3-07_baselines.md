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

## 실제 val(40건) 평가 결과 [확인]

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
