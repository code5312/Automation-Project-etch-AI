# 데이터 카드 (최종, W8-03)

- 버전: v1 (N=400, W3-02 생성 + W3-11/W3-12 버그 수정 반영)
- 작성일: 2026-11-09 / 최종화: 2026-11-14 (W8-03)
- 상태: **최종** — §4 "외부 배포본이 구버전" 항목은 재업로드하지 않기로 결정함
  (2026-11-14, 사용자 결정). 외부 배포본은 구버전으로 영구히 남으며, 아래 경고를
  최종 상태로 유지한다.
- 표기: **[확인]** 실행으로 검증됨 / **[제안]** 설계 값(검증 전) / **[미확인]** 근거 없는 기본값

## 1. 요약

ViennaPS(개방형 2D 공정 시뮬레이터)로 생성한 **가상(합성) 식각 공정 시뮬레이션 데이터**다.
양산 설비에서 수집한 데이터가 아니며, 특정 반도체 업체의 실제 모델·수치를 재현하지 않는다
(정의서 §2·§6). 용도는 대리 모델(MeshGraphNet/FNO)의 학습·평가·회귀 게이트 검증이다.

## 2. 생성 방법 [확인]

- 시뮬레이터: **ViennaPS 4.5.0 + ViennaLS 5.8.5**(ViennaLS 5.9.0은 import 실패, 사용 금지).
  400건 전체의 `meta.json["viennaps"]` 값이 모두 `"4.5.0"`으로 일치함을 확인.
- 시나리오: 2D 트렌치 식각 1종(SF6/O2, Si 기판 + 마스크). 정의서 §5.1.
- 샘플링: `scipy.stats.qmc.Sobol`(scramble=True, seed=20261027), 7차원 파라미터 Θ 공간에서
  준난수 추출(`src/etchpipe/sim/sampling.py`). train/val/test는 한 번의 추출을 순서대로 잘라
  교집합 0, OOD는 별도 분포(학습 범위 밖, 아래 참조).
- 시뮬레이션 설정(`configs/sim/base.yaml`, W3-01 동결): 스텝 상한 K=20, 조기종료 기준
  `depth_max=50µm`, `rays=300`, `grid_delta=0.3µm`, `step_min=1.0`분/스텝.
  (`rays`·`grid_delta`는 정확도-속도 트레이드오프 검증 스파이크 없이 seed 코드 기본값을
  그대로 채택 — §4의 한계 참조 [미확인].)
- 실행 인프라: kind(로컬 Kubernetes) Indexed Job, `parallelism=2`, `cpu=6`/잡. 400건 생성에
  약 53.6시간 소요(`reports/evidence/W3-02_gen_complete.md`).
- 출력: run당 `meta.json`(파라미터·시드·소요시간 등) + `snapshots/t*.vtp`(시간 스텝별 2D 표면
  선분 메시, `MaterialIds`·`Normals` 포함). 스냅샷 개수는 K+1(조기종료 시 actual_k+1)이고
  점 개수는 스냅샷마다 다르다.

## 3. 규모·분할 [확인]

| 구분 | 레시피 수 | 용도 | run_id 접두사 |
|---|---|---|---|
| train | 300 | 학습 | `r0000`~`r0299` |
| val | 40 | 검증 | `r0300`~`r0339` |
| test | 40 | 최종 평가 | `r0340`~`r0379` |
| OOD | 20 | 학습 범위 밖 평가(각 파라미터를 범위 밖 1.5배까지 확장) | `ood0000`~`ood0019` |
| golden | 10(위 train의 부분집합) | 회귀 게이트(고정, 변경 금지) | train과 동일 run_id |

- 실측 집계(400건 전체, `data/splits.json` 기준):
  - `wall_sec`: 최소 77.4초 / 평균 982.2초 / 최대 15287.0초
  - `final_depth_um`: 최소 0.6 / 평균 39.9 / 최대 64.8
  - 조기종료(`stopped_early=true`, `depth_max` 도달): 400건 중 194건(48.5%) — 종료 판정이
    스텝 완료 후 이루어져 `depth_max`(50)를 다소 넘는 값(최대 64.8)도 나올 수 있음 [확인:
    현상, 설계 의도와 일치].
  - 그래프 변환 후 점 수 최대치(그래프 크기 예산): **663**(`reports/evidence/W3-05_data_validation.md`).

### 3.1 입력 파라미터 Θ (7차원) [확인: 범위는 코드(`sampling.SPACE`) 기준, 물리적 타당성은 [제안]]

| 이름 | ViennaPS 파라미터 | 범위 | 스케일 |
|---|---|---|---|
| ion_flux | `ionFlux` | 3 ~ 30 | log |
| etchant_flux | `etchantFlux` | 1350 ~ 13500 | log |
| oxygen_flux | `passivationFlux` | 240 ~ 2400 | log |
| ion_mean_energy | `Ions.meanEnergy` | 50 ~ 200 | linear |
| ion_exponent | `Ions.exponent` | 100 ~ 1000 | log |
| mask_width | `MakeTrench(trenchWidth)` | 6 ~ 14 µm | linear |
| mask_height | `MakeTrench(maskHeight)` | 6 ~ 14 µm | linear |

OOD는 각 파라미터를 위 범위의 아래쪽 또는 위쪽으로 1.5배까지 확장한 구간에서 별도 샘플링
(`make_ood_recipes`, 동전 던지기로 방향 결정·seed로 결정적).

### 3.2 골든셋 [확인, W3-06]

train split 300건 중 SPACE 정규화 좌표에서 공간 중심까지의 거리로 10건을 선정(중심5·경계3·
극단2, `select_golden_recipes`). **train에서 선정했으므로 일반화 평가가 아니라 고정 회귀
기준점이 목적**이다 — val/test/OOD처럼 "모델이 안 본 데이터"가 아님을 명확히 한다.
`golden_hash=e95bdca4de55`(`reports/evidence/W3-06_golden_set.md`).

## 4. 알려진 이슈·한계

- **2D 한정** [확인]: 3D·KMC/MD 멀티스케일은 범위 제외(정의서 §5.3).
- **합성 데이터** [확인]: 가상 시뮬레이션 결과이며, 실제 생산 설비의 계측값과 대조·검증한 바
  없다. 모델의 유효 범위는 위 §3.1 파라미터 범위 안으로 한정된다.
- **`rays`/`grid_delta` 미검증** [미확인]: 정확도-속도 트레이드오프 스파이크(설계서 §14
  S0~S6)에 원래 포함되지 않았던 항목이라 seed 코드 기본값을 그대로 썼다. 재검토 트리거:
  ETL 표면 노이즈·부호 SDF 이상 또는 W4 롤아웃 오차 과다 시(`configs/sim/base.yaml` 주석).
- **OOD `ion_mean_energy` 음수 버그(W3-12, 수정 완료)** [확인]: 학습 범위 하한 확장 계산이
  음수 에너지를 뽑던 버그가 있었다. 영향받은 12건을 수정 코드로 재생성·로컬 백업했다
  (`reports/evidence/W3-12_ood_negative_energy_bugfix.md`).
- **SDF 내부 재료 경계 버그(W3-11, 수정 완료)** [확인]: ETL 단계의 부호 있는 거리장 계산
  버그. raw 데이터 자체에는 영향 없음, ETL 재실행으로 해결(`reports/evidence/W3-11_sdf_internal_interface_bugfix.md`).
- **외부 배포본(GitHub Release/Hugging Face)이 구버전** [확인, 영구적 한계로 확정]:
  W3-12 수정 전에 올린 `etch-raw-n400.zip`이 그대로 남아 있다. 즉 외부에 공개된 데이터셋에는
  OOD 12건이 수정 전(그중 5건은 물리적으로 불가능한 음수 에너지) 상태로 포함돼 있다.
  **재업로드하지 않기로 결정**(2026-11-14, 사용자 결정) — 로컬 `data/raw/`(이 저장소 기준)
  만 수정 후 버전이고, 외부 배포본은 구버전으로 영구히 남는다. 외부 배포본을 쓰는 사람은
  이 카드의 경고를 반드시 확인해야 한다(§6에도 동일 경고).
- **검증 범위**: `scripts/validate_data.py`(W3-05)는 스냅샷 개수·NaN/Inf·점 수>0·깊이
  비감소만 확인한다. 물리 정확도(ViennaPS 자체의 시뮬레이션 신뢰도)는 검증 대상이 아니다.

## 5. 검증 결과 [확인]

```
PYTHONPATH=src python scripts/validate_data.py data/raw
400 runs checked, 0 with errors
max_n_points (그래프 크기 예산) = 663
```

전체 400건에 스냅샷 개수 불일치·NaN/Inf·점 수 0·깊이 역행 없음(`reports/evidence/W3-05_data_validation.md`).

## 6. 라이선스·출처 [확인]

- 시뮬레이터 ViennaPS는 GPL-3.0 — 저장소 코드(Apache-2.0)와 별도로 `NOTICE`에 조건 표기
  (W1-05, 사용자 결정 2026-10-27).
- raw 데이터(400건, 81MB)는 git에 커밋하지 않는다(CLAUDE.md). 백업: 로컬
  `data/raw/` + GitHub Release(`data-v1-n400` 태그) + Hugging Face Datasets
  (`code5312/etch-ai-automation-data`) — 단, §4의 "외부 배포본이 구버전" 항목 참조.

## 7. 처리(ETL) 산출물 [확인]

- `dataset_hash=4b8c0ce1bc98`(W3-12 수정분 포함 N=400 전체로 `process_dataset` 실행,
  `reports/evidence/W3-04_stats_dataset_hash.md`).
- `stats.json`: train split(300건) 기준 node/edge/SDF 채널별 평균·표준편차.

## 8. 참고

- ADR-0001(인터페이스 동결), ADR-0003(부호 있는 SDF 규약)
- `reports/evidence/W3-0{1,2,3,4,5,6}_*.md`, `W3-11_*.md`, `W3-12_*.md`
