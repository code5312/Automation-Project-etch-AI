# 프로젝트 체크리스트: 식각 공정 AI 대리 모델의 검증·재학습 자동화 파이프라인

- 기준: `project-definition.md` v0.1(초안), `project-design.md` v0.1(초안)
- 항목 ID: `W<주차>-<번호>`. 참조: `FR-xx`(설계서 §3 요구사항 목록), `S0~S6`(설계서 §14), `D§`(설계서 장)
- 표기: **[확인]** 실행 검증됨 / **[제안]** / **[미확인]**

## 사용 규칙

- 항목 완료 = 코드 + 테스트 + CI 통과 + **증거 한 줄**(실행한 명령과 출력 요약)을 항목 끝에 기록. 증거 없으면 체크하지 않는다.
- 동시 진행 최대 2개. 스파이크는 시간 상자를 넘기면 대안으로 전환한다.
- 증거 파일은 `reports/evidence/`에 저장해 커밋한다. 파일명 규칙: `W<주차>-<번호>_<설명>.md|json|log`
- 정의서·설계서와 현실이 어긋나면 코드를 억지로 맞추지 말고 문서 수정 항목(`DOC-`)으로 기록한다.
- 선택 범위(정의서 5.2)는 **증거 ①②가 모두 확보되기 전에는 착수하지 않는다.**

## 핵심 증거 추적

| 증거 | 합격 기준 | 위치 | 주차 | 상태 |
|---|---|---|---|---|
| ① 게이트 차단 | 정상 통과, 노이즈·1에폭 모델 차단, 해시·골든 변조 차단(코드 2) | `reports/evidence/` + CI `gate-negative` 로그 | 2(더미), 5(실모델) | [ ] |
| ② 재학습 완주 | 모니터→생성→변환→학습→게이트→승격, 무개입 **자동 반복(최대 4회, ADR-0002)** 포함 완주, 전/후 오차 | `reports/evidence/retrain_*.json` | 6 | [ ] |

---

## 1주차: 스파이크와 뼈대

목표: 프로젝트가 돌아간다는 것과 위험 항목의 판정을 확보한다.

### 뼈대
- [x] W1-01 저장소 생성(공개), 설계서 §6 폴더 구조, `docs/`에 문서 3개, 루트 `CLAUDE.md`, `seed/`에 기존 참고 코드 이동 — 증거: `reports/evidence/W1-01_repo_skeleton.md`. `git init`+`origin` 설정(커밋·푸시는 안 함). seed/의 실제 참고 코드 11개를 역할별로 이동(C1→`src/etchpipe/sim/`, C2→`src/etchpipe/etl/`, 스모크 스크립트→`scripts/`, k8s 매니페스트→`deploy/k8s/`). 기존 `deploy/k8s/sim-job.yaml`이 local 변형 내용으로 잘못 들어가 있던 것을 올바른 파일로 교체.
- [x] W1-02 `.gitignore`(`data/`, `runs/`, `registry/`, `reports/` 중 evidence 제외, `*.vtp` 중 `golden/`·`tests/fixtures/` 제외), 시크릿 스캔(pre-commit) — 증거: `reports/evidence/W1-02_gitignore_secrets.md`. `detect-secrets scan --all-files` 결과 0건(컨테이너 실행).
- [x] W1-03 환경 3종 의존성 고정 파일(`requirements-sim/train/serve.txt`), 설치 재현 확인 [확인된 조합 기준] — 증거: `reports/evidence/W1-03_{sim,train,serve}_install.md`. 컨테이너 클린 설치로 3종 모두 import 성공. sim은 OS 라이브러리(`libgl1` 등) 추가 필요를 발견해 `docker/Dockerfile.sim`에 반영함. 설계서 §6에 따라 pyvista(ETL용)는 sim이 아니라 train 쪽으로 재배치하고 재검증.
- [x] W1-04 GitHub Actions `pr-check`(린트, 단위 테스트, `import viennaps` 스모크, 금지 표현 검사) — 증거: `reports/evidence/W1-04_pr_check_dryrun.md`. 4개 잡 모두 로컬 컨테이너 드라이런 통과(ruff, pytest 6건, viennaps import, 금지표현검사 — 1차 실패 1건 실제로 잡아내 README 수정). **S0 판정은 포함하지 않음**: 실제 GitHub Actions 실행(push) 전이라 미확정 — 별도 W1-S0로 남김.
- [x] W1-05 라이선스 파일: 저장소 코드 Apache-2.0, `docker/Dockerfile.sim`(ViennaPS 포함) 환경은 GPL-3.0 조건 NOTICE에 별도 표기 [사용자 결정 2026-10-27, 근거는 정의서 §11], 의존성 라이선스 표 작성 — 증거: `reports/evidence/W1-05_license.md`, `LICENSE`, `NOTICE`.

### 스파이크 (각 시간 상자 안에서 판정표 작성: `W1-S_<번호>_*.md`)
- [x] W1-S0 CI에서 ViennaPS·PyG·torch_scatter 설치 성공 — 증거: `W1-S0_ci_install.md`. 실제 GitHub Actions(run 40468419349) 4개 잡 모두 success.
- [x] W1-S1 레시피 20개 실행: 소요 시간 분포(최소/평균/최대, 코어 수), 깊이 분포 → `depth_max`·입력 범위·총 레시피 수(N) 결정 — 증거: `W1-S1_timing.md`. K=2 축소 측정(시간상자), wall 10.7~252.2s(평균 76.3s), 깊이 0.50~22.92µm. depth_max=50µm 제안, N은 2주차 정지조건 구현 후 재검증 필요.
- [x] W1-S2 같은 레시피·시드 2회 실행 → 스냅샷 일치 여부(불일치 시 비결정 요소·허용오차 정의) — 증거: `W1-S2_repro.md`. 점 개수·좌표·깊이 완전 일치(오차 0).
- [x] W1-S3 모델 A 롤아웃 사전 시험: 레시피 3~5개, 소규모 과적합 후 다단계 예측이 발산하는지 확인 — 증거: `W1-S3_rollout.md`. 완만한 선형 드리프트(폭발적 발산 아님), 원인(미수렴 학습·엣지 고정) 명확 → 진행 가능(조건부).
- [x] W1-S4 Docker 설치 확인 → `Dockerfile.sim` 빌드 → kind 클러스터 → `sim-job-local.yaml` 파드 4개 `Completed` → 파드 1개 삭제 후 재시도·이어하기 확인 — 증거: `W1-S4_kind.md`. 4/4 Completed, 파드 삭제 후 자동 재생성+skip 확인됨. cpu 리소스 과소 할당(1차 시도, 시간상자 초과로 중단) 발견·수정(cpu 1→6).
- [x] W1-S5 `MaterialIds` 0/10의 재료 매핑 확정 → `configs/etl/materials.yaml` — 증거: `W1-S5_materials.md`. **0=mask, 10=Si로 확정**(seed `etl.py` 주석은 반대로 적혀 있었음, 주석만 틀리고 실제 연산은 정상).
- [x] W1-S6 FNO 모듈 경로·시그니처를 설치본 소스에서 확인, 128×256 입력 forward/backward 1회, CPU 1에폭 시간 측정 — 증거: `W1-S6_fno.md`. `physicsnemo.models.fno.FNO`, forward 0.975s/backward 0.662s. torchvision 핀 없으면 import 자체가 깨짐을 확인.

**게이트 G1** (모두 Yes 시 2주차 진행)
1. 환경이 클린 상태에서 재현되는가 (CI 포함)
2. 레시피 N건 생성이 **5~7일 안에** 가능한가 (아니면 N 축소: 400→200)
3. S2 판정이 났는가 (일치 또는 허용오차 정의)
4. 롤아웃 사전 시험의 방향이 있는가 (S3: 진행 가능 / 후퇴 필요)
5. kind 또는 `xargs` 중 병렬 실행 경로가 확정됐는가

---

## 2주차: 워킹 스켈레톤

목표: **더미 모델**로 전체 흐름을 끝까지 돌리고 인터페이스(파일 형식·해시·종료 코드)를 고정한다.

- [x] W2-01 `run_sim`·`sim_batch`를 `etchpipe/sim/`으로 이식, `meta.json` 스키마(설계서 §3 C1), 정지 조건 구현 → FR-01, FR-03 — 증거: `src/etchpipe/sim/run_sim.py`. depth_max=50µm 정지조건 실제 시뮬레이션으로 확인(고flux 레시피가 depth_max=5로 1스텝에서 조기종료).
- [x] W2-02 `sampling`·`make_splits`(레시피 단위, 시드 고정) → FR-02 / 테스트: 같은 시드 동일 결과, 분할 교집합 0 — 증거: `tests/unit/test_sampling.py`(7건 통과). OOD는 학습범위 밖에서 별도 샘플링.
- [x] W2-03 CI 픽스처: 레시피 3개, K=3 `tests/fixtures/raw/` 생성·커밋 — 증거: 실제 ViennaPS 실행(r0000/r0001/r0002, 시드7, 109.2/23.7/56.6초), `tests/fixtures/raw/` 116KB.
- [x] W2-04 ETL 최소판: 그래프 변환, `dataset_hash`, `manifest.json` → FR-04, FR-05 (SDF는 3주차) — 증거: `tests/unit/test_etl.py`(9건). MaterialIds 주석 오류(W1-S5) 수정 반영.
- [x] W2-05 더미 모델 2종: 변위 0 예측(champion 역할), 무작위 변위(망가진 모델 역할) — 증거: `tests/unit/test_dummy_models.py`(5건).
- [x] W2-06 **지표 단위 테스트 먼저 작성 후** 지표 구현(Chamfer, depth) → 설계서 §3 C4 / 테스트: 정답=예측이면 0, 평행이동·직선·사각 트렌치 값, 순서 불변 — 증거: `tests/unit/test_metrics.py`(6건, 평행선 Chamfer=2d 해석값 검증 포함).
- [x] W2-07 `gate` 최소판(항목 1·4 + 종료 코드 0/1/2), 레지스트리 최소판(승격·`champion` 포인터·`history.jsonl`) → FR-10, FR-11 — 증거: `tests/unit/test_gate.py`(8건)+`test_registry.py`(4건). 종료코드 2는 dataset_hash·golden_hash 둘 다 독립 체크.
- [x] W2-08 서빙 최소판: `/health`, `/model`, `/predict`(더미) + API 키 필수 기동 → FR-12 — 증거: `tests/unit/test_serve.py`(6건). `ETCH_API_KEY` 없으면 import 시점에 기동 거부 확인.
- [x] W2-09 `pipeline-smoke` 워크플로: 픽스처 → ETL → 더미 학습(또는 로드) → 평가 → 게이트 → 레지스트리 → 서빙 호출 → FR-16 — 증거: `scripts/run_pipeline_smoke.py` 로컬 실행 성공(exit 0), `.github/workflows/pr-check.yml`에 `pipeline-smoke` 잡 추가.
- [x] W2-10 `gate-negative` 워크플로(더미): 정상 통과, 무작위 변위 모델 차단, `dataset_hash` 불일치 코드 2 → **증거 ① 프로토타입** — 증거: `scripts/run_gate_negative.py` 3개 케이스 모두 기대대로 동작(exit 0), CI 잡 추가.
- [x] W2-11 인터페이스 동결: 스키마·해시 규칙·종료 코드를 문서에 확정, 변경 시 ADR — 증거: `docs/adr/0001-interface-freeze.md`.

**게이트 G2**: `pipeline-smoke`와 `gate-negative`가 CI에서 초록이다. 이후 단계는 부품 교체이며 인터페이스를 바꾸지 않는다.

---

## 3주차: 실데이터와 변환

목표: 본 데이터 생성을 **백그라운드로 먼저 시작**하고, 그동안 ETL을 완성한다.

- [ ] W3-01 S1 결과로 `configs/sim/base.yaml` 확정(`depth_max`, 범위, K, rays), `make_recipes`로 레시피 N건, `splits.json`(train/val/test/ood/golden) — 1차 확정: `configs/sim/base.yaml`(2026-11-03). N=410(300/40/40/20/10) 시간예산 [확인, `W3-01_probe8_depthmax_kind.md`, parallelism=2 기준 약 2.7일]. `rays`·`grid_delta`는 정확도-속도 트레이드오프 검증 없이 [미확인] 상태로 보류 — 재검토 트리거: W3-03 ETL에서 표면 노이즈/부호 SDF 이상, 또는 W4 롤아웃 오차가 과도할 때. `make_recipes`/`make_splits` 실행 완료(2026-11-03) — 증거: `reports/evidence/W3-01_recipes_splits.md`(`scripts/gen_splits.py`, `data/recipes.json` 400건, `data/splits.json` train300/val40/test40/ood20, probe8 r0000~r0007 파라미터 불변 확인). 아직 남음: 골든셋(10) 선정 규칙(중심5·경계3·극단2) 알고리즘 결정 — ADR-0001 동결 스키마상 `splits.json`에 golden 키 추가 안 함.
- [x] W3-02 본 데이터 생성 시작(kind Job 또는 `xargs -P`), 진행률·실패율 요약 스크립트, 실패 건 재시도
  — 완료(2026-11-05) 증거: `reports/evidence/W3-02_gen_complete.md`(40/40 shard 전부 `Succeeded`,
  실패 0건, 400/400 run `meta.json` 완전성 확인, 로컬·GitHub Release·HF 3중 백업).
  `sim_progress.py`는 Job 완료 후(Running 파드 0개) 상태를 처리하지 못하는 한계 발견 — 후속 과제로 기록.
- [x] W3-03 ETL 완성: **부호 있는 SDF**(닫힌 폴리곤 안/밖), y 범위 자동 결정, 재료 매핑 적용 → FR-04 / 테스트: 직선·사각 트렌치에서 부호·값, 보잉 형상 안/밖, NaN 없음 — 완료(2026-11-03) 증거: `reports/evidence/W3-03_signed_sdf.md`(실제 경계선은 닫힌 폴리곤이 아닌 열린 체인임을 확인해 even-odd 규칙으로 구현, `PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q` → 14 passed, 실제 fixture로 사후 확인해 NaN 없음·시간에 따라 solid 비율 감소 확인). y 범위는 `configs/sim/base.yaml`의 depth_max=50/`ood.extend`에서 `auto_yr()`로 자동 계산(옛 하드코딩 `(-30,12)`는 depth_max=50 기준으로 바닥이 잘리는 버그였음). 재료 매핑은 W2-04에서 이미 완료 확인. 부호 규약·y범위 공식은 ADR-0001 §6이 ADR 작성을 요구한 항목이라 `docs/adr/0003-signed-sdf-convention.md`로 동결(승인, 사용자 결정 2026-11-03).
- [x] W3-04 정규화 통계 `stats.json`(**train에서만**), `dataset_hash` 생성, 같은 입력=같은 출력 해시 일치 테스트 → FR-05
  — 완료(2026-11-07) 증거: `reports/evidence/W3-04_stats_dataset_hash.md` (신규 테스트 4건
  포함 `test_etl.py` 20 passed, 전체 스위트 64 passed). 실제 N=400(버그 수정분 포함)에
  처음 `process_dataset` 실행 성공, `dataset_hash=4b8c0ce1bc98`, `stats.json`
  node/edge/sdf 채널별 평균·표준편차 확인.
- [x] W3-05 데이터 검증 스크립트: 모든 run에 K+1 스냅샷, NaN 없음, 점 수>0, 깊이 단조 증가, 점 수 최대치 보고(그래프 크기 예산)
  — 완료(2026-11-07) 증거: `reports/evidence/W3-05_data_validation.md` (신규 테스트 5건
  포함 `test_etl.py` 25 passed, 전체 스위트 69 passed). 실제 N=400 전체 오류 0건,
  그래프 크기 예산(점 수 최대치)=663. `scripts/validate_data.py`.
- [x] W3-06 골든셋 확정: `golden/raw/` 10개 복사, `golden.json`, `golden_hash`, 변경 시 테스트 실패
  — 완료(2026-11-08) 증거: `reports/evidence/W3-06_golden_set.md`. 선정 규칙(중심5·경계3·
  극단2)을 SPACE 정규화 좌표의 공간 중심까지 거리로 구체화(`select_golden_recipes`),
  train split에서 10건 선정·복사, `golden_hash=e95bdca4de55`. 신규 테스트 7건 포함
  `tests/unit` 76 passed. 변조 시 테스트 실패를 수동으로도 확인(`params_hash` 변경 →
  AssertionError, 복구 후 재통과).
- [x] W3-07 베이스라인 2종: (a) 변위 0, (b) 레시피→깊이 MLP 후 직선 외삽, val 지표 기록
  — 완료(2026-11-08) 증거: `reports/evidence/W3-07_baselines.md`
  (+`W3-07_baseline_metrics.json`). 둘 다 최종 표면 1장 예측(t=0→actual_k) 기준 평가.
  신규 테스트 5건 포함 `tests/unit` 81 passed. 실제 val(40건): (a) Chamfer 평균 14.464/
  깊이오차 평균 40.964µm, (b) Chamfer 평균 8.552/깊이오차 평균 2.397µm.
- [x] W3-08 데이터 카드 초안(생성 방법·범위·알려진 한계·시뮬레이터 버전)
  — 완료(2026-11-09) 증거: `docs/data_card.md`. ViennaPS 4.5.0+ViennaLS 5.8.5, Sobol
  샘플링, N=400 실측 집계(wall_sec/final_depth_um/조기종료 194건), 파라미터 범위 표,
  golden 10건, W3-11/W3-12 버그 수정 이력과 **GitHub Release·HF가 아직 구버전**임을
  한계로 명시. 금지 표현 검사 통과(grep 0건). W8-03에서 최종본으로 갱신 예정.
- [x] W3-09 (계획 외 추가, 코드 리뷰로 발견) ETL 성능: `process_dataset`에 run_id 단위
  `ProcessPoolExecutor` 병렬화(`max_workers`, 기본값 1=기존과 동일, 회귀 없음) 추가.
  (E,N) 완전 벡터화는 실측상 6배 역효과(0.148s→0.928s)라 기각하고 레시피 단위
  병렬화만 적용 — 완료(2026-11-03) 증거: `reports/evidence/W3-09_etl_parallel.md`
  (`PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q` → 15 passed, 직렬/병렬
  dataset_hash·sdf 배열 동일 확인).
- [x] W3-10 (계획 외 추가, 코드 리뷰로 발견) 서빙 API 키 비교를 `hmac.compare_digest`로
  교체(`!=` short-circuit 비교의 타이밍 공격 이론적 가능성 제거) — 완료(2026-11-03)
  증거: `reports/evidence/W3-10_api_key_timing.md`(틀린 키 401·맞는 키 200 재확인,
  Windows 소켓 환경 이슈는 이번 변경과 무관한 기존 이슈로 별도 기록).
- [x] W3-11 (계획 외 추가, 사용자 제보로 발견) SDF 내부 재료 경계 버그 수정:
  `addInterfaces=True`로 내보낸 메시의 solid-solid 내부 경계(마스크가 덮은 Si 상단)가
  외곽 경계와 MaterialIds로 구분되지 않아 짝수-홀수 교차 판정이 깨지는 버그를
  `shapely` polygonize+union 기반 필터(`_filter_internal_interfaces`)로 수정 —
  완료(2026-11-06) 증거: `reports/evidence/W3-11_sdf_internal_interface_bugfix.md`
  (기존 15건+신규 회귀 테스트 1건 = 16 passed, 전체 스위트 59 passed, 실제 fixture
  수정 전/후 비교로 버그 재현·해결 확인). `data/processed/`가 비어있어 실 데이터
  영향 없음 확인. 새 의존성 `shapely==2.2.0` 추가(사용자 확인 후).
- [x] W3-12 (계획 외 추가, 사용자 제보로 발견) OOD `ion_mean_energy` 음수 버그 수정:
  `make_ood_recipes`의 linear-scale 하한 확장(`lo - span`)이 `ion_mean_energy`에서
  음수 구간을 포함해(`lo=50 < span=75`) 물리적으로 불가능한 음수 에너지를 샘플링하던
  버그를 `max(lo-span, 1e-6)` 클램프로 수정 — 완료(2026-11-07) 증거:
  `reports/evidence/W3-12_ood_negative_energy_bugfix.md` (신규 회귀 테스트 1건 포함
  8 passed). 영향받은 ood 12건(`ood0004,07,08,09,10,12,13,14,15,17,18,19`) 재생성·
  로컬 백업 완료. GitHub Release·Hugging Face는 아직 수정 전 버전 — 재업로드는
  후속 단계. +100건 생성 계획은 split 비율 변경(DOC- 필요)이라 보류(사용자 확인).

**게이트 G3**: train/val/test/ood가 로드되고, 데이터 검증이 통과하며, 베이스라인 지표가 있다. 생성이 덜 끝났어도 현재 완료분으로 `v1`을 고정하고 진행한다.

---

## 4주차: 모델 A 학습과 롤아웃 판정

**역할 분담 결정(2026-11-09)**: MeshGraphNet·FNO의 **학습·오차율 확인·재확인**은 GPU 가용 팀원이
담당(아래 **[GPU 팀원]** 표기 항목). 그 외(모델 초안을 파이프라인에 매핑, 전체 흐름 1회 구성·연결)는
CPU로 계속 진행(**[CPU/매핑]** 표기). 사유: CPU 컨테이너에서 MeshGraphNet 1 epoch(레시피 5개,
전이쌍 71개)에 150~290초 소요(호스트 CPU 경쟁 포함) — 30 epoch 과적합 테스트조차 비효율적임을
실측 확인(`scripts/_overfit_probe_tmp.py`, 아래 W4-03 참고).

팀원이 전달한 MGN 초안(`runs/mgn_draft/ckpt_mgn.pt`, epoch=5, processor_size=**12**)·FNO
초안(`runs/fno_draft/`, train 300·val 40 기준 val Chamfer 중앙값 0.118µm·깊이오차 0.92% —
이미 실전급)을 분석해 입력 정규화 누락을 발견·수정(`train/loop.py`의
`NodeEdgeNorm`/`load_norm_stats`, 2026-11-09 — `W4-02_chamfer_loss_train_loop.md` 추가
참고). GPU 팀원은 학습 시 `train_one_epoch(..., norm=load_norm_stats("data/processed/<hash>/stats.json"))`로
불러서 쓸 것. **processor_size 정리 완료**: 체크포인트 실측값 12로
`MeshGraphNetModel`·`configs/model/mgn_v1.yaml` 기본값을 맞춤(이전 제안값 6에서 변경,
2026-11-09). **FNO 격자 y범위 정리 완료**: `etl.auto_yr()`을 팀원 모델 기준 −68~21로 변경
(아래 W6-11 메모 참고) — 기존 `data/processed/4b8c0ce1bc98/sdf/`는 이전 범위(−53~21)로
생성된 것이라 재생성 필요[미확인: 재생성 일정].

- [x] W4-01 학습 스크립트: YAML 설정 한 파일, 시드·설정·코드 SHA·패키지 버전을 `runs/`에 저장 → FR-06
  — 완료(2026-11-09) 증거: `reports/evidence/W4-01_train_script_skeleton.md`.
  `src/etchpipe/train/run_train.py`(`init_run`), `configs/model/mgn_v1.yaml`. 새 의존성
  `PyYAML==6.0.3` 추가(사용자 확인 후). 신규 테스트 5건 포함 `tests/unit` 86 passed.
  실제 설정 파일로 실행해 `runs/<run_id>/{config.yaml, git_sha, packages.json}` 생성 확인.
- [x] W4-02 손실 Chamfer 구현·테스트, 학습 루프(전이쌍, 팀포싱)
  — 완료(2026-11-09) 증거: `reports/evidence/W4-02_chamfer_loss_train_loop.md`.
  `src/etchpipe/train/chamfer_loss.py`(미분 가능, `torch.cdist`), `train/loop.py`
  (`iter_transition_pairs`, `train_one_epoch`). 신규 테스트 8건 포함 `tests/unit` 94
  passed. 토이 모델로 루프 검증(실제 MeshGraphNet 연결은 W4-03, 호스트에
  nvidia-physicsnemo 미설치라 이번엔 확인 못 함).
- [ ] W4-03 **소부분집합 과적합 테스트**(레시피 5개에서 손실이 0에 가깝게) — **[GPU 팀원 이관,
  2026-11-09]**. CPU(`etch-train:0.1` 컨테이너, `--memory=2g`)에서 실제 MeshGraphNet 래퍼
  (`src/etchpipe/models/meshgraphnet.py`, processor_size=6/hidden=64)로 시도: lr=1e-3일 때
  손실이 0.8635→0.7560(epoch 0~8)에서 plateau, 0에 가깝지 않음 확인. lr=1e-2로 재시도 중
  (epoch 0: loss=1.0815, epoch당 150~290초, 호스트 CPU 경쟁으로 변동) 비효율 판단해 중단·이관.
  이관 자료: `scripts/_overfit_probe_tmp.py`(임시 프로브, 미커밋), `docker/Dockerfile.train`
  (`etch-train:0.1`, 빌드·import 검증됨), `data/processed/4b8c0ce1bc98/graphs/`(전체 400 run
  변환 완료).
- [ ] W4-04 본 학습(모델 `processor_size=12, hidden=64` — GPU 팀원 초안 체크포인트 실측값에
  맞춤, 2026-11-09 변경), 에폭당 시간 기록, 예산 3시간 이내 [제안] — **[GPU 팀원]**
- [ ] W4-05 롤아웃 v1(연결 유지, 법선·엣지 재계산, 재료 이전 값 유지): K스텝 예측, val 오차 기록
  — **[CPU/매핑] 완료(2026-11-09)**: `src/etchpipe/train/rollout.py`(`rollout`,
  `rollout_val_error`, `recompute_edge_features`, `recompute_normals`) 구현, 토이 모델로
  연결·재료·theta 불변·t 증가를 직접 대조 검증 + 실제 fixture로 `rollout_val_error` 플러밍
  확인 — 증거: `reports/evidence/W4-05_rollout_v1.md`(신규 테스트 6건 포함 `tests/unit` 93
  passed). **[GPU 팀원]** 실제 학습된 체크포인트(`processor_size=12`) 기준 val 40개
  오차 기록은 W4-04 완료 후 `rollout_val_error(..., norm=load_norm_stats(...))`로 채움.
- [ ] W4-06 롤아웃 오차 vs 베이스라인(a)(b) 비교표 — **[CPU/매핑]** 비교표 스크립트는 지금 준비,
  **[GPU 팀원]** 실제 수치는 학습 결과 반영 후 채움.
- [ ] W4-07 실패 시 롤아웃 v2(호길이 재샘플링) 시도 (시간 상자: 1.5일) — **[GPU 팀원]** (D4 판정 이후,
  필요 시)

**판정 D4 (주 말)**: val 최종 스텝 Chamfer가 베이스라인 (b)보다 작으면 **통과**. 아니면 **후퇴**:
- 후퇴안: **최종 프로파일 1장 예측** — 입력 (초기 표면, Θ) → 출력 최종 표면. 학습 쌍을 (0, K)로 바꾸고 롤아웃 제거. 게이트·평가는 최종 스텝 Chamfer 기준으로 그대로 사용 [제안]
- 후퇴 시 `DOC-` 항목으로 정의서 §5.1·설계서 §3 C3의 롤아웃 서술을 수정하고 ADR 작성
- 어느 쪽이든 증거 ①②는 영향 없음

**게이트 G4**: 모델 A 체크포인트 1개가 val 평가를 통과한 상태로 존재한다.

---

## 5주차: 평가·게이트·레지스트리 (증거 ①)

- [ ] W5-01 지표 전체 구현: IoU, CD_top/mid/bot, taper, bowing, latency + 스텝 5·10·20 보고 → FR-07 / 단위 테스트 전부 통과
- [ ] W5-02 `eval.py`: 체크포인트 + `dataset_hash` → `metrics.json`, 그림 PNG(예측 vs 정답 오버레이), 결정적, 하드웨어·시드 기록
- [ ] W5-03 OOD 20개 평가 → 별도 표
- [ ] W5-04 게이트 임계값 산출: 첫 정상 모델 val 중앙값 Chamfer × 1.5 → `configs/pipeline/gate.yaml` 고정(변경 시 ADR) → 설계서 §3 C5
- [ ] W5-05 게이트 완성: 항목 1~4, 챔피언 없음 케이스, `gate_result.json`
- [ ] W5-06 **부정 테스트 6종**(설계서 §3 C5): 정상 통과 / 노이즈 모델 차단 / 1에폭 모델 차단 / 시뮬레이터 버전 변경 데이터로 학습한 모델 차단 / `dataset_hash` 불일치(코드 2) / 골든 변조(코드 2)
- [ ] W5-07 레지스트리 완성: `manifest.json`, `valid_range.json`, 자동 모델 카드, 승격·롤백·`history.jsonl` → FR-11
- [ ] W5-08 CI `gate-negative`를 실모델 기준으로 교체, 로그를 아티팩트로 저장
- [ ] W5-09 **증거 ① 제출본**: 부정 테스트 6종 결과를 `reports/evidence/W5-09_gate_negative.md`로 정리·커밋

**게이트 G5**: 증거 ①이 CI 로그와 evidence 파일로 존재한다. 이 시점에 일정이 1주 이상 밀렸으면 6주차의 FNO 범위를 줄인다.

---

## 6주차: 재학습 루프 (증거 ②), 모델 B, 서빙

우선순위 순서로 진행한다. 시간이 모자라면 아래에서부터 줄인다.

### ① 재학습 루프 (먼저)
- [ ] W6-01 `configs/sim/live.yaml`(드리프트 오버라이드 `Substrate.A_ie` ×1.15), `sim_config_id` 반영
- [ ] W6-02 모니터: 신규 레시피 10개를 live 설정으로 시뮬레이션 → champion 예측과 Chamfer 비교 → `drift_threshold`(= champion 골든 Chamfer × 1.5) 초과 시 **자동 반복 루프**(ADR-0002) 트리거 → FR-09
- [ ] W6-03 드리프트 데이터 60개 생성(train 40 / val 10 / golden_drift 10) → ETL → 새 `dataset_hash` — 루프 1차 시도의 입력
- [ ] W6-04 **자동 반복 루프 구현(최대 4회, ADR-0002)**: 1차 champion에서 3~5에폭 미세조정 → 게이트. BLOCK이면 2차(드리프트 train 40→100개 자동 확장 재생성, ViennaPS 추가 호출) → 게이트. BLOCK이면 3차(에폭 10~15) → 4차(학습률 조정) 순서로 자동 진행. 어느 시도든 PASS면 즉시 승격하고 루프 종료, 4회 모두 BLOCK이면 champion 유지 + 알림. 시도마다 게이트 결과 기록
  — **[CPU/매핑]** 루프 오케스트레이션 코드(시도 순서·게이트 호출·기록)는 CPU에서 구현 가능.
  **[GPU 팀원]** 실제 미세조정(에폭 실행) 자체는 GPU에서 수행.
- [ ] W6-05 `python -m etchpipe.retrain run` 단일 명령으로 전 과정(에스컬레이션 루프 포함) 무개입 실행, `reports/evidence/retrain_<ts>.json`에 `attempts: [{attempt, config, gate_result}, ...]` 기록
- [ ] W6-06 **증거 ② 제출본**: 승격까지 완주한 실행 1회 이상(자동 반복 과정 포함) + 재학습 전/후 오차 표 + 시도별 로그. 4회 모두 BLOCK이면 그 자체를 결과로 기록(사람이 사후 원인 분석, 이번 루프의 무개입 완주 정의는 깨지 않음)
- [ ] W6-07 `retrain-demo` 수동 워크플로(소규모 설정)

### ② 서빙 (다음)
- [ ] W6-08 `/predict` 실모델 연결(champion 로드, CPU), 입력 검증, 범위 경고(`in_range`), 2배 초과 422, 키 인증, 요청 크기·속도 제한 → FR-12, FR-13
- [ ] W6-09 오류 응답 정제(경로·스택 미노출), 이상 입력 100개 퍼징에서 5xx 없음
- [ ] W6-10 `docker-compose.yml`, 로컬 직접 실행 문서화, 두 방식 모두 `/health` 확인, 지연 측정(조건 병기)

### ③ 모델 B (마지막, 줄일 수 있음)
- [ ] W6-11 FNO 구현(S6 결과 기반), SDF 0등고선 복원 → FR-08 — **[CPU/매핑]** 모델 래퍼·입출력
  매핑 코드(`src/etchpipe/models/fno.py`, MeshGraphNet 래퍼와 같은 패턴)는 CPU에서 작성.
  **[GPU 팀원]** 실제 학습(원래 계획 "CPU 학습(격자 128×256)"에서 변경, GPU로 진행)·오차율 확인.
- [ ] W6-12 모델 A/B 비교표(같은 test 분할·지표·조건)
- [ ] W6-13 시간이 모자라면 모델 B를 "단일 조건 단순 비교"로 축소하고 비교표는 유지, 축소 내용을 문서에 기록

**게이트 G6**: 증거 ②가 존재하고, 서빙이 로컬·Compose 모두에서 동작한다.

---

## 7주차: 대시보드, 통합, 점검

- [ ] W7-01 대시보드 화면 3개(레시피 입력→예측, 예측 vs 정답·모델 비교, 지표·게이트 이력), Plotly.js 파일 포함, API만 호출 → FR-14
- [ ] W7-02 **클린 환경 재현**: 새 clone + 새 컨테이너에서 README만 따라 스모크 파이프라인 실행, 막힌 곳 전부 기록·수정 → FR-17
- [ ] W7-03 실패 시나리오 테스트: 시뮬레이션 실패, 데이터 손상, 모델 로드 실패, 게이트 오탐/미탐
- [ ] W7-04 보안: 의존성 취약점 스캔, 시크릿 스캔, `torch.load` `weights_only` 적용 확인, 커밋 이력에 데이터·키 없음 확인
- [ ] W7-05 전체 테스트·커버리지 실행, 깨진 테스트 0
- [ ] W7-06 금지 표현 검사 통과(코드·README·대시보드·모델 카드), 정의서 6장 기준 → FR-18
- [ ] W7-07 전체 파이프라인 단계별 실행 시간 측정, 병목 1개 개선
- [ ] W7-08 문서·코드 불일치 목록 정리 후 문서 수정

**게이트 G7**: 클린 환경 재현 성공, 테스트 전부 통과, 금지 표현 0건.

---

## 8주차: 문서, 발표, 예비

- [ ] W8-01 최종 벤치마크 표(정확도·지연·OOD 거동), 조건·하드웨어·시드 병기
- [ ] W8-02 한계 분석: 2D 한정, 가상 데이터, 유효 범위, 보잉·언더컷 구간 오차, 미확인 항목 처리 결과
- [ ] W8-03 README(설치 → 데모 한 줄 실행 → 증거 ①② 재현 방법), 데이터 카드·모델 카드 최종
- [ ] W8-04 ADR 정리, 정의서 대비 달성/미달 표
- [ ] W8-05 발표 자료(비전공자용 1장 요약 포함), 예상 질문과 답변
- [ ] W8-06 데모 대본 5분(정상 통과 → 망가진 모델 차단 → 드리프트 재학습 → 대시보드), 리허설 2회, 백업 영상 녹화
- [ ] W8-07 과장 표현 점검(정의서 6장), 향후 과제 정리
- [ ] W8-08 태그 `v1.0`, 불필요 파일 정리
- [ ] 예비: 밀린 항목 처리. **선택 범위(정의서 5.2)는 증거 ①②가 확보된 경우에만**

**게이트 G8**: README·발표·태그·백업 영상·증거 파일이 모두 있다.

---

## 주간 루틴

- 월: 이번 주 항목 선택(동시 진행 2개 이하), 스파이크는 시간 상자 지정
- 매일: 3줄 로그(한 것 / 막힌 것 / 다음) → `reports/log/YYYY-MM-DD.md`
- 금: 게이트 판정 Yes/No, 근거 없는 체크 해제, 2분 화면 녹화, ADR 필요 결정 기록, 다음 주 위험 상위 3개

## 후퇴 기준 요약

| 상황 | 대응 |
|---|---|
| G1에서 생성 시간 초과 | N 400→200, 격자·rays 축소 |
| 깊이 폭주 | 범위 축소 또는 `depth_max` 정지 |
| D4 롤아웃 실패 | 최종 프로파일 1장 예측으로 후퇴(W4 참조) |
| kind 불가 | `xargs -P` 대체 경로(코드 동일) |
| FNO CPU 학습 느림 | 격자 64×128, 그래도 어려우면 단순 비교로 축소 |
| 재학습 승격 실패(루프 1차 BLOCK) | 자동 반복 루프(ADR-0002)가 드리프트 train 100개 확장→에폭 증가→학습률 조정 순으로 자동 처리. 4회 모두 BLOCK이면 champion 유지 + 사람이 사후 분석 |
| CI 설치 지연 | 캐시, 사전 빌드 이미지 |
| 전체 지연 | 6주차 ③ → 7주차 대시보드 화면 수 → 선택 범위 순으로 축소 |

## 최종 산출물 체크

- [ ] 파이프라인 코드, README, Docker·Compose
- [ ] 데이터셋 생성 스크립트와 데이터 카드
- [ ] 모델 비교 보고서와 모델 카드
- [ ] CI 설정과 증거 ①② 로그 (`reports/evidence/`)
- [ ] 서빙 API, 대시보드, 백업 영상
- [ ] ADR, 한계·향후 과제, 발표 자료