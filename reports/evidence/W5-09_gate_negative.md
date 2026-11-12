# W5-09: 증거 ① 제출본 — gate-negative 부정 테스트 6종 (실모델 기준)

- 날짜: 2026-11-12
- 코드: `scripts/run_gate_negative.py`(W2-10 더미 모델판을 실모델로 교체, W5-06/W5-08)
- 게이트 임계값: `configs/pipeline/gate.yaml`(W5-04)
- 데이터: `golden/raw/`(git에 커밋된 10개 레시피) — `etl_run`으로 그 자리에서 그래프 변환.
  `data/raw`·`data/processed`는 `/data/` 전체가 `.gitignore`에 걸려 CI에서 못 쓴다.
- 체크포인트: `runs/mgn_tuned/model.pt` 1개만 `.gitignore` 예외로 커밋함([확인]
  2026-11-12 사용자 결정 — `/runs/`·`*.pt` 블랭킷 규칙에 대한 유일한 예외).
  `runs/20261110T020341143621_c07d1b/ckpt_last.pt`(1에폭 케이스의 실제 체크포인트)는
  커밋 안 함 — 로컬에선 그대로 쓰고, 없으면(CI) 합성 대체로 넘어간다.

## 실행

```
PYTHONPATH=src python scripts/run_gate_negative.py
```

## 결과 [확인] — 6종 모두 기대대로 동작

| # | 케이스 | 모델 | 기대 | 실제 | 판정 |
|---|---|---|---|---|---|
| 1 | 정상 통과 | `runs/mgn_tuned/model.pt` | passed=True | passed=True, exit=0 | OK |
| 2 | 노이즈 모델 차단 | `MeshGraphNetModel`(무작위 가중치, 미학습) | passed=False | passed=False, exit=1(깊이/CD 오차 85.6%) | OK |
| 3 | 1에폭(저학습) 모델 차단 | `runs/20261110T020341143621_c07d1b/ckpt_last.pt`(epoch=6, 로컬 실측) | passed=False | passed=False, exit=1(깊이/CD 오차 6.8%) | OK |
| 4 | 시뮬레이터 버전 변경 모델 차단 | 합성 대체(아래) | passed=False | passed=False, exit=1(깊이/CD 오차 96.0%, 롤아웃 비율 2.52) | OK |
| 5 | `dataset_hash` 불일치 | — | exit=2 | exit=2 | OK |
| 6 | 골든셋 변조 | — | exit=2 | exit=2 | OK |

```
gate-negative 통과: 6개 케이스 모두 기대대로 동작
```

## 알려진 한계·합성 대체 [미확인 → 합성, 명시]

- **케이스 4(시뮬레이터 버전 변경)**: 실제로 그 체크포인트가 없다 — ViennaLS 5.9.0은
  import 자체가 안 돼(CLAUDE.md 환경 제약) 그 버전으로 학습하는 게 불가능하다.
  `SyntheticDepthBiasModel`로 mgn_tuned의 실제 변위 중 깊이(y) 성분을 1.6배로 부풀려
  깊이 체계적 오차를 모사했다 — **실제로 그 데이터로 학습한 모델이 아니다**
  [제안, 2026-11-11 사용자 결정].
- **케이스 3(1에폭)의 CI 대체재**: 저장소에 체크포인트가 없는 환경(CI)에서는
  `NoisyDisplacementModel`(mgn_tuned 변위에 시드 고정 가우시안 노이즈 추가)로 대체한다.
  이 문서의 수치는 **로컬 실제 체크포인트** 기준이다 — CI에서 돌면 다른(합성) 수치가
  나온다.
- **CD(중간 선폭) 신호**: `check_cd_depth_error`가 깊이·CD를 각각 받지만,
  `rollout_val_error_by_step`은 아직 CD를 계산하지 않는다(`metrics.cd_at_y`/`cd_profile`은
  W5-01/02에서 추가됐지만 어떤 실행 경로도 호출 안 함). 이 스크립트는 `cd_pred`/
  `cd_true`에 `depth_pred`/`depth_true`를 복제해서 넘긴다 — 표에 적힌 "깊이/CD 오차"는
  실질적으로 깊이 판정 하나와 같다(독립적인 CD 검증 아님). CD 배선은 별도 항목으로 남김.
- **표본 크기**: golden 10개 레시피 기준이다(val-40 전체가 아님) — CI 실행 시간 예산과
  golden/raw가 git에 커밋돼 있다는 점 때문. val-40 전체 기준 수치는
  `reports/evidence/W4-06_rollout_vs_baseline.md`/`W5-03_ood_eval.md`에 이미 있다.
- **정상 모델 케이스의 표본 민감도(실측, 발견)**: 처음에 val split 앞 5개로 테스트했을 때
  "정상 모델"이 롤아웃 후반 급증 규칙(④, `rollout_max_ratio=2.0`)에서 비율 2.035~2.804로
  **임계값을 살짝 넘겨 BLOCK**됐다(스텝을 5,10에서 5,10,15,20으로 늘리자 2.804→2.035로
  줄었지만 여전히 넘김). golden 10개로 늘리자 비율이 임계값 아래로 내려가 깨끗이
  PASS했다 — 5개짜리 표본이 노이즈가 커서 경계선에 걸렸던 것으로 보인다[확인, 실측].
  이 규칙이 실제 체크포인트 데이터로 끝까지 계산된 건 이번이 처음이다(W5-05/07은
  더미 모델로만 테스트됨).

## 부수 발견·수정 (이번 항목 작업 중)

- **CI `unit-tests`/`gate-negative` job에 torch가 없었다**(W4-01부터, 2026-11-12에야
  발견) — `pyproject.toml` dependencies와 `requirements-dev.txt` 둘 다 torch를 안
  설치하는데 `test_rollout.py` 등 6개 테스트 파일이 모듈 최상단에서 `import torch`를
  했다. `.github/workflows/pr-check.yml` 두 job 모두에 `requirements-train.txt` 설치
  단계를 추가했다[확인, 사용자 결정 2026-11-12]. **이 발견 이전에 W4-01~W5-07로 표시된
  "CI 통과" 증거는 전부 로컬 `etch-train:0.1` 컨테이너 기준이었고, 실제 GitHub Actions로는
  한 번도 검증되지 않았었다** — `backend-security` 브랜치가 아직 push/merge 안 돼 지금까지
  드러나지 않음.
- `train/rollout.py::rollout_val_error_by_step`의 각 스텝 결과에 `depth_pred`/
  `depth_true`(원값)를 추가했다 — `check_dimension_error`가 절대오차가 아니라 정답
  크기 대비 %오차를 보려면 필요.
- `gate/gate.py::run_gate`가 `golden_pairs=[]`일 때 ①을 건너뛰도록 수정(이전엔
  `ZeroDivisionError`) — 더미 모델 전용 `model.predict(pos)` 인터페이스에 맞는 실모델
  골든 어댑터가 아직 없어서 실모델 테스트는 ①을 비워 둔다(참고 지표라 passed엔 무영향).

## 테스트

```
python -m ruff check src/ tests/ scripts/ → All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
  → 163 passed, 3 failed(기존 git 미설치 환경 문제, test_run_train.py, 본 세션 변경과 무관)
```
