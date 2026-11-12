# W5-03: OOD 20개 평가 ([CPU/매핑])

- 날짜: 2026-11-11
- 코드: `src/etchpipe/eval/eval.py`에 `split` 인자 추가(`val`/`test`/`ood` 선택, 기존
  `main()`은 `val` 하드코딩이었음)
- 체크포인트: `runs/mgn_tuned/model.pt`(ep2, W4-04 완료 체크포인트), 마스크고정+교차가드+
  고리제거 켬(val-40 평가와 동일 설정 — `reports/evidence/W4-06_rollout_vs_baseline.md`)

## 실행

```
PYTHONPATH=src python -m etchpipe.eval.eval runs/mgn_tuned/model.pt \
  --config configs/model/mgn_v1.yaml --out-dir reports/eval/mgn_tuned_ood20 \
  --device cpu --steps 5,10,20 --n-plots 2 --fix-mask --guard --loop-removal --split ood
```

## 결과 [확인] — val 대비 "OOD 오차 비"(설계서 §3 C4)

| | val(40) | OOD(20) | OOD/val 비율 |
|---|---|---|---|
| Chamfer 평균(µm) | 0.373 | **2.154** | **5.8배** |
| Chamfer 중앙값(µm) | 0.388 | 1.265 | 3.3배 |
| 깊이오차 평균(µm) | 1.126 | **13.971** | **12.4배** |
| 깊이오차 중앙값(µm) | 0.905 | 11.427 | 12.6배 |
| 선 꼬임(합) | 0 | 0 | - |

## 해석 [제안]

- OOD에서 오차가 뚜렷하게(3~13배) 커지는 건 예상된 방향이다 — 학습 범위 밖 입력에서
  성능이 떨어지는 건 정상이고, 오히려 val/OOD가 비슷했다면 분할이 제대로 안 됐다는
  신호였을 것이다.
- 깊이오차가 Chamfer보다 훨씬 더 많이 나빠진다(12배 vs 3~6배) — OOD 레시피에서 전체
  모양(Chamfer)보다 식각 **깊이** 자체를 더 크게 틀린다는 뜻. 레시피 Θ(특히 깊이에 큰
  영향을 주는 채널)가 학습 범위를 벗어났을 때 깊이 추정이 가장 먼저 깨지는 것으로 보임.
- 선 꼬임은 OOD에서도 0건 — 마스크고정+교차가드+고리제거가 입력 분포가 달라져도
  안정적으로 동작함을 보여준다(물리적으로 불가능한 형상은 여전히 안 나옴).
- OOD 레시피 20개 자체가 `sampling.make_ood_recipes`로 학습 범위 밖에 생성된 것이라,
  이 수치가 "유효 범위 밖에서 모델을 믿으면 안 된다"는 근거가 된다 — W6-08 서빙의
  `in_range` 경고 로직과 바로 연결된다.

## 테스트

```
python -m ruff check src/etchpipe/eval/eval.py → All checks passed
PYTHONPATH=src python -m pytest tests/unit/test_eval.py -q → 8 passed
```

## 다음 연결 지점

- `in_range` 서빙 경고(W6-08)는 이 OOD/val 오차 비 수치를 근거로 "범위 밖 입력은 오차가
  3~13배 커질 수 있다"고 문서화할 수 있다.
