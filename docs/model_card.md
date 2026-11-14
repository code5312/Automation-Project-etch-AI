# 모델 카드

- 작성일: 2026-11-14 (W8-03)
- 표기: **[확인]** 실행으로 검증됨 / **[제안]** 설계 값(검증 전) / **[미확인]** 검증 전, 사실로 쓰지 않음
- 두 모델 모두 **가상(ViennaPS) 시뮬레이션 데이터로만 학습**했다. 양산 설비에서 수집한
  데이터가 아니며 특정 업체의 모델·수치를 재현하지 않는다(정의서 §2·§6,
  [`docs/data_card.md`](data_card.md)).

## 1. 현재 배포 모델(레지스트리 champion) [확인]

| 항목 | 값 |
|---|---|
| `model_id` | `mgn_retrain_20261112T212130` |
| 모델 종류 | 모델 A, MeshGraphNet (PhysicsNeMo) |
| `dataset_hash` | `26a29dcca7da` |
| `git_sha` | 레지스트리 카드에 `[미확인: 재학습 루프 실행, git 명령 미사용]`로 자체 기록됨 — 재학습 루프가 git 명령을 쓰지 않아 비워짐(버그 아님, 설계상 공백) |
| `gate_result` | `passed: true, exit_code: 0` |
| 생성 시각 | `2026-11-12T21:21:30Z` |
| 아티팩트 | `runs/retrain/mgn_retrain_20261112T212130/model.pt`, SHA-256 `716821...580f50a` |

**유효 범위**(`valid_range`, 학습 데이터 실제 분포에서 계산 — 정의서 §7.1 표의 제안값과
근접하지만 1:1로 같지는 않음):

| 파라미터 | min | max |
|---|---|---|
| ion_flux | 3.126 | 29.612 |
| etchant_flux | 1425.202 | 13202.992 |
| oxygen_flux | 251.550 | 2248.187 |
| ion_mean_energy | 50.925 | 198.745 |
| ion_exponent | 106.019 | 939.256 |
| mask_width | 6.036 | 13.783 |
| mask_height | 6.063 | 13.925 |

출처: `registry/mgn_retrain_20261112T212130/card.json`(`Registry.register()`가 자동 생성).

**승격 이력**(`registry/history.jsonl`, 전체 2건): ① `mgn_tuned_ep2` 최초 승격
(2026-11-12 05:31) → ② 드리프트 재학습 1차 시도가 즉시 PASS해 `mgn_retrain_20261112T212130`
으로 교체(2026-11-12 21:21, 증거②: [`W6-06_retrain_evidence.md`](../reports/evidence/W6-06_retrain_evidence.md)).

## 2. 모델 A vs 모델 B 최종 비교 [확인, 기존 실측 재인용 — 새 계산 없음]

전체 벤치마크 표와 측정 조건(파이프라인별 Chamfer 정의 차이 포함)은
[`reports/evidence/W8-01_final_benchmark.md`](../reports/evidence/W8-01_final_benchmark.md)
참고. 요지만 인용:

| 지표(골든10 기준) | 모델 A(`mgn_tuned` ep2) | 모델 B(`fno_tuned`) |
|---|---|---|
| IoU | 0.9689 | 0.9745 |
| Chamfer(µm) | 0.0901 | 0.0693 |
| 깊이오차(%) | 0.8058 | 0.346 |
| 판정 | 합격 | 합격 |

- **중요한 차이**: 모델 A는 위 §1처럼 **자동화된 C5 게이트·C6 레지스트리 승격 플로우를
  실제로 통과**해 champion이 된 적이 있다. 모델 B(FNO)는 **그 자동 플로우를 통과한 적이
  없다** — 위 골든10 수치는 팀 자체 채점 노트북(`runs/fno_tuned/metrics/`)으로 평가한
  것이고, `registry/history.jsonl`에 FNO 항목은 전혀 없다[확인, `grep -i fno
  registry/*.jsonl registry/manifest.json` 결과 0건]. 두 모델을 "같은 틀에서 비교"한다는
  목표(정의서 §5.1)는 지표 수준에서는 달성했지만, **게이트 통과·레지스트리 승격까지
  실제로 거친 것은 모델 A뿐**이다.
- 비교가 공정하지 않을 수 있는 지점: 위 표의 모델 A는 ep2(저학습) 체크포인트다 — 학습
  성숙도 차이로 모델 B가 유리하게 보일 수 있다(`W6-12_model_comparison.md` 캐비엇).

## 3. 알려진 한계 [확인, 중복 작성 대신 W8-02 인용]

전체 한계 분석(2D 한정, 가상 데이터, 유효 범위, **보잉·언더컷 구간 오차 — 핵심 공백**,
미확인 항목 처리 결과)은 [`reports/evidence/W8-02_limitations.md`](../reports/evidence/W8-02_limitations.md)
에 있다. 모델 카드 관점에서 가장 중요한 세 가지만 다시 강조한다:

1. **보잉·언더컷 구간에서 모델이 얼마나 틀리는지 실측치가 없다** — `eval/metrics.py::bowing()`이
   실제 모델 결과에 호출된 적이 0건(함수 자신의 docstring이 명시). 언더컷은 전용 지표도
   없고, 롤아웃 구현 자체가 "언더컷 제외"를 가정한다(`train/rollout.py`).
2. **FNO(모델 B)는 OOD(학습 범위 밖) 평가가 전혀 없다** — 모델 A만 OOD 20개로 평가됨
   (Chamfer val 대비 ×5.8, 깊이오차 ×12.4, `reports/evidence/W5-03_ood_eval.md`).
3. **추론 지연은 end-to-end(API 전체) 19~20초만 실측**됐고, 모델 단독 forward 시간·
   ViennaPS 대비 속도비는 구현만 있고 실제로 계산·보고된 적이 없다.

## 4. 참고

- [`docs/data_card.md`](data_card.md) — 학습에 쓰인 데이터의 생성 방법·분할·한계.
- [`reports/evidence/W8-01_final_benchmark.md`](../reports/evidence/W8-01_final_benchmark.md) — 정확도·지연·OOD 전체 표.
- [`reports/evidence/W8-02_limitations.md`](../reports/evidence/W8-02_limitations.md) — 한계 분석 전문.
- [`docs/adr/0002-auto-retry-retrain-loop.md`](adr/0002-auto-retry-retrain-loop.md),
  [`docs/adr/0005-recipe-pass-rate-gate-and-stat-tests.md`](adr/0005-recipe-pass-rate-gate-and-stat-tests.md) — 게이트·재학습 판정 근거.
