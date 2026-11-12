# W6-12: 모델 A/B 비교표 (같은 test 분할·지표·조건)

- 날짜: 2026-11-12
- 코드: `scripts/run_model_comparison.py`(신설)
- 모델 A: `runs/mgn_tuned/model.pt`(epoch=2, processor_size=6, hidden_dim_processor=128,
  fix_mask+guard 켬)
- 모델 B: `runs/fno_tuned/`(GPU 팀원 전달, `FNOModel` 래퍼 경유)
- 분할: `data/splits.json`의 `test`(40개, r0340~r0379) 전체
- 정답(true): 두 모델 다 같은 기준 — `outer_path`로 추출한 실제 최종 스텝 표면선
  (모델 B가 자체 SDF 0등고선으로 정답을 다시 뽑으면 추출 방식이 달라 불공정해짐)

## 실행

```
PYTHONPATH=src python scripts/run_model_comparison.py
```

## 결과 [확인, 실측] — test 40개 전체

| | Chamfer 평균(µm) | 깊이오차 평균(µm) |
|---|---|---|
| 모델 A (MeshGraphNet, mgn_tuned) | 0.408 | 1.540 |
| 모델 B (FNO, fno_tuned) | 0.155 | 0.320 |

전체 40개 레시피 결과: `reports/eval/W6-12_model_comparison.json`.

## 해석 [제안]

- 모델 B(FNO)가 Chamfer·깊이오차 둘 다 모델 A(MeshGraphNet)보다 뚜렷하게 낫다
  (Chamfer 약 2.6배, 깊이오차 약 4.8배 차이). 이 격차는 **같은 조건 비교라기보다
  학습 성숙도 차이**로 보는 게 맞다 — 모델 A(`mgn_tuned`)는 `epoch=2`(W4-04 초기
  체크포인트)인 반면, 모델 B(`fno_tuned`)는 40에폭 + 2스텝 롤아웃 학습까지 마친
  "v2" 완성판이다(README 자체 평가: 골든10 Chamfer 0.069µm, 합격).
- 모델 B의 자체 보고값(README: test 40 Chamfer 0.101)과 이 비교표의 값(0.155)이
  정확히 같지 않다 — "정답" 추출 방식이 다르기 때문이다(이 비교표는 `outer_path`
  메시 폴리라인, README는 팀 골든 채점 노트북의 SDF 기반 방식). 방향과 크기는
  일치한다(둘 다 "매우 낮음" 수준) — 절대값 차이는 측정 방법론 차이로 설명됨[제안].
- 모델 A를 더 학습된 체크포인트(예: 100에폭 완주한 `runs/20261110T...b369f6`)로
  바꾸면 격차가 줄어들 가능성이 있다 — 이 비교는 **현재 registry에 등록된 champion
  기준**이라는 점을 명시해 둔다.

## 테스트

```
python -m ruff check scripts/run_model_comparison.py → All checks passed
```
`run_model_comparison.py`는 실제 체크포인트·실제 데이터가 필요해 단위 테스트는 없음
(`run_sim.py`/`run_baselines.py`와 같은 관례 — 이 실행 자체가 증거).
