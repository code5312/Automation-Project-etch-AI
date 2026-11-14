# W8-01: 최종 벤치마크 표 (정확도·지연·OOD 거동)

이 문서의 모든 수치는 기존에 실제로 실행·기록된 증거 파일에서만 가져왔다(새로 계산한
값 없음). 각 표에 조건·하드웨어·시드를 함께 적는다. 같은 지표 이름(예: Chamfer)이라도
**측정 파이프라인이 다르면 섞지 않고 표를 나눈다** — 아래 §0 참고.

## 0. 측정 파이프라인 3종 (섞으면 안 되는 이유)

| 파이프라인 | 산출 | Chamfer 정의 |
|---|---|---|
| A. `eval.py`(guarded rollout) | `reports/eval/mgn_tuned_val40_guarded/`, `.../mgn_tuned_ood20/` | 자체 조건(`fix_mask`,`guard`,`loop_removal`) 포함 롤아웃, 스텝별 |
| B. 팀 골든셋 채점 노트북 | `runs/mgn_tuned/metrics/champion.json`, `runs/fno_tuned/metrics/champion.json`, `runs/fno_tuned/metrics/FNO_round2_golden_summary.csv` | 노트북 자체 정의(코드 수정 없이 팀원이 작성) |
| C. `scripts/run_model_comparison.py` | `reports/evidence/W6-12_model_comparison.md` | outer_path 폴리라인 기준 |

이 셋은 독립적으로 만들어진 결과라 숫자가 서로 다를 수 있다(실제로 모델 A의
test-40 Chamfer가 B에서는 0.378, C에서는 0.408로 다름 — 둘 다 실측치이며, 어느 쪽도
오류로 지우지 않는다. 파이프라인이 다르다는 사실만 병기한다)[확인].

## 1. 정확도 — 골든 10 (파이프라인 B, 모델 A/B 직접 비교 가능)

| 지표 | 모델 A (`mgn_tuned` ep2) | 모델 B (`fno_tuned`) |
|---|---|---|
| IoU | 0.9689 | 0.9745 |
| F1 | 0.9842 | 0.9870 |
| Chamfer(µm) | 0.0901 | 0.0693 |
| 깊이오차(%) | 0.8058 | 0.346 |
| 중간CD오차(%) | 3.0073 | 1.703 |
| 바닥CD오차(%) | 4.8614 | 2.045 |
| 롤아웃 후반÷전반 | 1.9134 | 1.722 |
| 선 꼬임/물리위반 | 0건 | 0건 |
| 판정 | 합격(모델 고유 검사 미확인) | 합격 |

출처: `runs/mgn_tuned/metrics/champion.json`, `runs/fno_tuned/metrics/FNO_round2_golden_summary.csv`.
모델 B가 모든 지표에서 더 낫다 — 단 모델 A는 ep2(저학습)라 학습 성숙도 차이로 해석해야
한다[확인, `docs/project-checklist.md` W6-12 캐비엇].

## 2. 정확도 — val-40 / test-40

### 2a. 파이프라인 A (모델 A만, guarded rollout, 스텝별)

| 스텝 | Chamfer 평균/중앙값 | 깊이오차 평균/중앙값(µm) |
|---|---|---|
| 5 | 0.1293 / 0.1102 | — |
| 10 | 0.2155 / 0.1919 | — |
| 20 | 0.3178 / 0.3247 | — |
| final | 0.3729 / 0.3876 | 1.1257 / 0.9054 |

n=40(val), 선 꼬임 0건. 출처: `reports/eval/mgn_tuned_val40_guarded/metrics.json`.
조건: `checkpoint=runs/mgn_tuned/model.pt`, `dataset_hash=4b8c0ce1bc98`, `seed=0`,
`fix_mask=guard=loop_removal=true`. 하드웨어: CPU(Linux/WSL2), `cuda_available=false`,
패키지 `torch 2.14.1+cpu / torch_geometric 2.8.1 / torch_scatter 2.1.2 / physicsnemo 2.2.2`.

### 2b. 파이프라인 C (모델 A vs B, test-40, 같은 그라운드트루스)

| 모델 | Chamfer | 깊이오차 |
|---|---|---|
| A(`mgn_tuned` ep2) | 0.408 | 1.540 |
| B(`fno_tuned`) | 0.155 | 0.320 |

출처: `reports/evidence/W6-12_model_comparison.md`(`scripts/run_model_comparison.py`).
모델 B가 뚜렷하게 낫지만, 모델 A가 ep2뿐이라 공정한 성숙도 비교는 아니다[확인,
동일 캐비엇].

### 2c. 파이프라인 B, 모델 B의 val/test (모델 A와 직접 비교 불가 — 파이프라인 다름)

| 구분 | IoU | Chamfer | Chamfer(last) | 깊이오차(%) | 중간CD오차(%) |
|---|---|---|---|---|---|
| val | 0.9584 | 0.0964 | 0.1554 | 1.2123 | 2.6672 |
| test | 0.9549 | 0.1014 | 0.1552 | 1.1441 | 2.0300 |

출처: `runs/fno_tuned/metrics/champion.json`(`test_evaluated_once="2026-11-10 23:12"`).

## 3. 베이스라인 대비 (모델 A, val-40, 파이프라인 A 초기 버전)

| 모델 | Chamfer 평균 | 깊이오차 평균(µm) |
|---|---|---|
| zero-displacement(무변화) | 14.464 | 40.964 |
| MLP+선형외삽 | 8.552 | 2.397 |
| MeshGraphNet(최종, §2a final) | 0.373 | 1.126 |

출처: `reports/evidence/W3-07_baselines.md`, `W3-07_baseline_metrics.json`.

## 4. OOD 거동 (20개 레시피)

| 모델 | Chamfer 평균/중앙값 | 깊이오차 평균/중앙값(µm) | val 대비 |
|---|---|---|---|
| A(`mgn_tuned` ep2) | 2.1542 / 1.2655 | 13.9707 / 11.4268 | Chamfer ×5.8, 깊이오차 ×12.4 |
| B(FNO) | [미확인] — OOD 20개 평가 기록 없음 | — | — |

A 출처: `reports/eval/mgn_tuned_ood20/metrics.json`(§2a와 동일 조건: `seed=0`,
`dataset_hash=4b8c0ce1bc98`, CPU), `reports/evidence/W5-03_ood_eval.md`. 선 꼬임 0건(모든
스텝). B는 저장소 전체를 검색했으나 OOD 레시피에 대한 평가가 어디에도 없다 — 실행하지
않은 것을 사실로 쓰지 않는다[미확인, W8-02 한계 분석에 반영 필요].

## 5. 추론 지연

| 구분 | 값 | 조건 |
|---|---|---|
| `/predict` end-to-end(API) | **약 19~20초** | CPU, 2-step rollout, `docker-compose`와 `docker run` 양쪽에서 동일하게 측정 |
| 모델 단독 forward 시간(ViennaPS 메시 구성 제외) | [미확인] | `eval/metrics.py::measure_latency()` 구현·단위테스트는 있으나 실제 데이터로 호출된 적 없음(호출부 0건) |
| 추론 속도 비(ViennaPS 시간÷모델 시간, ADR-0005) | [미확인] | `eval/metrics.py::speed_ratio()` 동일 — 구현됐으나 실측 보고 없음 |
| `max_latency_s`(게이트 임계치) | [미확인, 비워둠] | `configs/pipeline/gate.yaml` — 미실측이라 설정 안 함 |

출처(유일한 실측): `reports/evidence/W6-10_docker_compose_local_run.md`(W6-10). 주의:
이 19~20초는 FastAPI 앱 전체(ViennaPS 메시 생성 + 모델 추론 + 후처리) 종단 시간이며,
모델 자체의 순수 forward 시간이 아니다[확인, 과장 방지].

## 6. 조건·하드웨어·시드 메타데이터 요약

- 서빙·테스트·평가: **CPU만**(`cuda_available=false`, 프로젝트 전역 제약).
- 모델 A 평가(§2a, §4): `seed=0`, `dataset_hash=4b8c0ce1bc98`,
  `torch 2.14.1+cpu / torch_geometric 2.8.1 / torch_scatter 2.1.2 / physicsnemo 2.2.2 / numpy 2.4.6 / scipy 1.17.1`,
  플랫폼 `Linux(WSL2)`. `git_sha` 필드는 해당 JSON 안에서 깨져(mojibake) 읽을 수 없음[미확인].
- 모델 A/B 학습: "GPU 팀원"이 수행(역할분담 결정, 2026-11-09)했으나 **구체적 GPU
  기종은 저장소 어디에도 기록돼 있지 않다**[미확인].
- 모델 B(FNO) 학습: champion run명 `E3_unroll2_noise0.1_seed0`(seed0, seed1 비교 후
  seed0 채택) — 그 외 하드웨어 기록 없음[미확인].

## 결론 — 이번 항목에서 메울 수 없는 미확인 목록

- FNO(모델 B)의 OOD 20개 거동.
- 모델 단독(순수 forward) 추론 지연, 추론 속도 비 — 구현은 있으나 미실측.
- `max_latency_s` 게이트 임계치.
- 두 모델 학습에 쓰인 GPU 기종.
- 모델 A 평가 JSON의 `git_sha` 필드(손상).
- 모델 A의 test-40 Chamfer가 파이프라인별로 다름(0.378 vs 0.408) — 오류가 아니라 서로
  다른 측정 정의이므로 §0처럼 구분해서만 인용할 것.

이 미확인 항목들은 W8-02(한계 분석)에서 "미확인 항목 처리 결과"로 다시 다룬다.
