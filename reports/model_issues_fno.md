# 모델 B(FNO) 문제·오차 기록

이 파일은 체크리스트 증거(`reports/evidence/`)와는 별도로, **미세조정 때 참고할 "지금
뭐가 안 맞는지"**를 모델 관점에서 계속 추적하는 용도다. MGN 쪽은 `reports/model_issues_mgn.md`.

## 현재 상태 (2026-11-10)

FNO는 아직 우리 쪽(또는 GPU 팀원)에서 실제로 학습을 돌리기 시작하지 않았다
(`docs/project-checklist.md` W6-11, `[GPU 팀원]` 담당, 아직 6주차 전이라 착수 전).
지금 있는 건 팀원이 넘겨준 **초안 체크포인트**(`runs/fno_draft/{best.mdlus, config.json,
norm.json, predict.py, README.json}`)뿐이고, 우리가 이 초안을 직접 평가·재실행한 적은
없다(MGN처럼 `run_rollout_comparison.py`로 독립 검증한 적 없음).

## 초안이 자체 보고한 수치 [미확인 — 우리 쪽에서 독립 검증 안 함]

`runs/fno_draft/README.json`(팀 내부용 문서) 기준:

- 모델: FNO(PhysicsNeMo 2.2.2), `main_s1`, 폭 16, 파라미터 591,721, 시드 20261027
- 입력 `(1, 8, 128, 256)`: 채널 0 = φ/10(SDF), 채널 1~7 = Θ(ion_flux 등, 일부 log+정규화,
  격자 전체에 같은 값 broadcast)
- 격자: y −68~21µm 128칸, x −15~15µm 256칸 — **ADR-0003 업데이트(2026-11-09)의 새
  y 격자 범위와 일치 확인됨**([확인], `docs/adr/0003-signed-sdf-convention.md`)
- 출력: `u`, 다음 φ = 현재 φ + 0.3×sinh(u), 롤아웃 시 `u`를 `[-0.5, asinh(50)]`로 클램프
- **val 성능(40런, 롤아웃 최종, 클램프+다시정리)**: Chamfer 중앙값 **0.118µm** / 깊이오차
  중앙값 **0.92%**(최악 3.8%) / CD 바닥오차 중앙값 **6.9%**
- 학습 데이터: train 300런만 사용, **val 40런은 에폭 고르는 데 사용**(MGN과 달리 이미
  val 기준 체크포인트 선택을 하고 있음 — `model_issues_mgn.md` 문제 ①과 대조)
- 팀원 본인 명시: "현재 개선 실험 진행 중"

## 열린 문제 (미해결)

아직 우리가 직접 학습·평가를 시작하지 않아 "발견한 문제"는 없다. W6-11 착수 시 이
섹션을 채운다. 미리 체크해 둘 사항(선제적 메모, 아직 검증 안 함):

- `config.json`/`norm.json`의 정규화 기준(채널별 평균·표준편차)이 우리 쪽
  `data/processed/<hash>/stats.json`과 같은 분할(train 300)로 계산됐는지 확인 필요.
- 입력 8채널 중 Θ 채널 7개가 "격자 전체에 같은 값"으로 broadcast되는 방식이 우리
  ETL(`etl.py`)의 SDF 출력 규약과 그대로 맞는지 실제 로드해서 확인 필요.
- `predict.py`가 우리 `src/etchpipe/models/fno.py`(아직 미작성, W6-11)와 같은
  `GraphDisplacementModel`류 인터페이스로 감쌀 수 있는 형태인지 확인 필요.

## 해결된 문제 (히스토리)

(아직 없음)
