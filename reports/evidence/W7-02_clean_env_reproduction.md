# W7-02: 클린 환경 재현

새 `git clone`(별도 디렉터리) + 새 venv에서 README를 그대로 따라가며 막힌 곳을 전부
기록했다. 실행한 환경: Windows, Python 3.14.0(로컬에서 실제 사용 가능한 버전 —
README/`pyproject.toml`은 3.11+).

## 재현 절차 (README Quick Start/Install/Testing 그대로)
```
git clone <repo> repo && cd repo && git checkout main
python -m venv .venv && .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pip install shapely==2.2.0
python scripts/run_pipeline_smoke.py
python scripts/run_gate_negative.py
python -m pytest tests/unit --ignore=tests/unit/test_baseline.py -q
ruff check .
python scripts/check_forbidden_expressions.py
```

## 통과한 것
- `pip install -e .` — 성공(단 `pyproject.toml`의 `project.license` TOML 테이블 표기가
  deprecated라는 setuptools 경고 발생, 2027-02-18까지 SPDX 문자열로 바꿔야 함 — 이번
  범위 밖으로 기록만).
- `python scripts/run_pipeline_smoke.py` — README 그대로 성공.
- `ruff check .` — 통과.
- `python scripts/check_forbidden_expressions.py` — 통과.

## 막힌 것 (실측, 전부 README 수정으로 해결)
1. **`pip install -U pip` 자체가 실패** — Windows에서 pip가 자기 자신 직접 업그레이드를
   거부(`python -m pip install -U pip`로 해야 함). README 명령을 그대로 쓰면 첫 줄부터
   에러. (README에는 반영 안 함 — pip/OS 자체의 일반적인 동작이라 이 저장소만의 문제가
   아님, 참고로만 기록.)
2. **`python scripts/run_gate_negative.py`가 dev 환경에서 즉시 크래시**
   (`ModuleNotFoundError: No module named 'torch'`). README Quick Start는 이 단계를
   "ViennaPS·PyTorch 없이 돌아가는 부분"이라고 명시했지만, W5-06/[ADR-0005](
   ../../docs/adr/0005-recipe-pass-rate-gate-and-stat-tests.md) 작업으로 이 스크립트가
   더미 모델이 아니라 실제 champion(torch+physicsnemo)을 쓰도록 완전히 바뀐 뒤로 더
   이상 사실이 아니었다. **README 수정**: train 환경이 필요하다는 경고와 실제 실행
   시간(13~15분)·결과(10개 케이스) 추가.
3. **`pytest tests/unit --ignore=tests/unit/test_baseline.py`도 여전히 깨짐** — README는
   torch가 필요한 파일이 `test_baseline.py` 하나뿐이라고 했지만 실제로는 7개 더
   (`test_chamfer_loss.py`, `test_eval.py`, `test_monitor.py`, `test_rollout.py`,
   `test_run_train.py`, `test_train_loop.py`, `test_train_main.py`) 있어 수집 단계에서
   전부 에러. **README 수정**: 8개 파일 전체 목록과 "전체 스위트는 train 환경에서"
   안내로 교체.
4. **`shapely` 트러블슈팅 항목이 이미 해결된 문제를 설명** — README는 "`shapely`가
   `pyproject.toml`에 없다"고 했지만 실제로는 이미 `pyproject.toml` 의존성에 들어가
   있다(`pip install -e .` 단계에서 이미 설치됨, 2026-11-13 수정된 것으로 보임).
   Quick Start의 별도 `pip install shapely==2.2.0` 줄도 실질적으로 불필요(아무 효과
   없는 중복). **README 수정**: 해당 트러블슈팅 항목과 Quick Start의 중복 줄 제거.
5. **README 본문 전체가 1~2주차 수준에서 멈춰 있었음** — Overview 컴포넌트 표가
   C3(학습)/C4(평가)/C7(서빙)/C8(재학습)/C9(대시보드)를 "미착수/최소판/골격"으로
   적고 있었는데, 실제로는 전부 구현·실행 증거가 있는 상태(MeshGraphNet+FNO, 전체
   지표 세트, ADR-0005 게이트, ADR-0002 재학습 루프, 대시보드 3화면). CI job 표도
   `serve-tests`가 빠져 있었고, 디렉터리 구조도 `retrain/`·`models/meshgraphnet.py`·
   `models/fno.py`·`dashboard/static/`·`docker/Dockerfile.train`·`.github/workflows/
   retrain-demo.yml` 등을 전혀 반영 못 하고 있었다. **README 수정**: 전면 재작성
   (`/github-readme-maker` 형식 적용) — Overview 표, 디렉터리 트리, Quick Start, API
   endpoints 신설 섹션, Option table, CI job 표, Trouble shooting 전부 현재 상태 기준
   으로 교체.

## 검증 (README 수정 후)
- `python scripts/check_forbidden_expressions.py` → 통과(exit 0).
- 이모지 포함 여부 직접 스캔 → 없음.
- README에 적은 경로(ADR 0001~0005, `docs/serving.md`, `configs/pipeline/gate.yaml`,
  `docker-compose.yml`, `tests/unit/test_golden.py`, `tests/unit/
  test_requirements_pinning.py` 등) 전부 실제 존재 확인.
- 내비게이션 앵커(`#quick-start` 등)가 실제 헤딩과 전부 일치하는지 확인.

## 미확인 / 남은 작업
- Linux/macOS 환경에서는 재현 안 해봄(Windows에서만 실측).
- `pyproject.toml`의 `project.license` TOML 테이블 deprecated 경고는 이번 범위 밖으로
  남겨둠 — 2027-02-18 전에 SPDX 문자열로 바꿔야 함(setuptools 경고 원문 참고).
- README 수정 후 CI(GitHub Actions)에서 `forbidden-expressions` job 재확인 필요.
