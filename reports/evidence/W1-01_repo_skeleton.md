# W1-01 증거: 저장소 구조

- 날짜: 2026-10-27
- 명령: `find . -maxdepth 3 -type d` (결과는 설계서 §6 폴더 구조와 1:1 대응하는 `configs/{sim,etl,model,pipeline}`, `src/etchpipe/{sim,etl,models,eval,gate,registry,serve,dashboard}`, `tests/{unit,gate,reproducibility}`, `scripts/`, `docker/`, `deploy/k8s/`(기존), `.github/workflows/`, `docs/`(기존), `reports/evidence/` 생성 확인)
- `git init` + `git remote add origin https://github.com/code5312/Automation-Project-etch-AI.git` 실행(이미 설정되어 있어 "remote origin already exists" — 기존 설정과 충돌 없음 확인). 커밋·푸시는 하지 않음(사용자 지시).

## seed/ 참고 코드 이식 (2차 — 실제 반영)

1차 작업 시점에는 `seed/`가 비어 있어 로직 없이 docstring만 작성했었다. 이후 사용자가 `seed/`에 실제 파일 11개(`Dockerfile.sim`, `etl.py`, `mgn_smoke_test.py`, `pvc.yaml`, `requirements.txt`, `requirements-train.txt`, `run_sim.py`, `sampling.py`, `sim_batch.py`, `sim-job.yaml`, `sim-job-local.yaml`)를 채워 넣었다. 내용은 수정하지 않고(`mv`) 역할에 맞게 이동했다:

| seed/ 원본 | 이동 위치 | 비고 |
|---|---|---|
| `run_sim.py`, `sampling.py`, `sim_batch.py` | `src/etchpipe/sim/` | C1. 내용 무수정 |
| `etl.py` | `src/etchpipe/etl/` | C2. 설계서 §6에 따라 ETL은 ViennaPS 환경이 아니라 PhysicsNeMo(train) 환경에 속함 — pyvista 의존성도 requirements-train.txt로 이동 |
| `mgn_smoke_test.py` | `scripts/` | pytest 테스트가 아니라 수동 실행용 진단 스크립트(print 기반, `data/processed/` 실데이터 필요) |
| `sim-job.yaml` | `deploy/k8s/sim-job.yaml` | **기존 파일을 덮어씀** — 기존 `deploy/k8s/sim-job.yaml`은 실제로는 "local" 변형 내용(`completions: 4`)이 `sim-job.yaml`이라는 이름으로 잘못 들어가 있었다. seed의 진짜 전체 스윕 버전(`completions: 40`)으로 교체 |
| `sim-job-local.yaml` | `deploy/k8s/sim-job-local.yaml` | 새 파일. 체크리스트 W1-S4가 명시하는 파일명과 일치 |
| `pvc.yaml` | (이동 안 함) | 기존 `deploy/k8s/pvc.yaml`과 바이트 단위로 동일 — 중복이라 그대로 뺐다 |
| `Dockerfile.sim` | `docker/Dockerfile.sim` | **수정함** — 이유는 아래 |
| `requirements.txt` | (통합) | 이미 검증된 루트 `requirements-sim.txt`로 대체(ViennaPS/ViennaLS 핀 동일, numpy/scipy는 이미 더 정확히 고정돼 있었음). pyvista는 뺐다(아래 참조) |
| `requirements-train.txt` | (통합) | 루트 `requirements-train.txt`에 pyvista/numpy/scipy 추가(ETL이 train 환경으로 이동했으므로) |

### Dockerfile.sim 수정 두 가지 (내용 보존 원칙의 예외, 이유 명시)

1. **apt 패키지 추가**: 원본은 `pip install`만 하는데, W1-03에서 이미 확인한 대로 `python:3.13-slim`에서는 `import viennaps`가 `libX11.so.6` 등 누락으로 실패한다(`reports/evidence/W1-03_sim_install.md`). `libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6`을 설치하는 `RUN apt-get install` 줄을 추가하지 않으면 이 이미지는 빌드는 되지만 실행 시 깨진다.
2. **COPY 경로 변경**: 원본은 `COPY src/ /app/src/`였는데, 저장소의 `src/`는 이제 `etchpipe` 패키지 전체(학습·서빙 코드 포함)를 담고 있다. `sim_batch.py`가 `from run_sim import run_sim`로 **평면(flat) import**를 쓰기 때문에(패키지 상대 import가 아님), 이 셋(`run_sim.py`/`sampling.py`/`sim_batch.py`)만 평면으로 복사해야 import가 깨지지 않는다. `COPY src/etchpipe/sim/ /app/src/`로 바꿔 `sim_batch.py`·`run_sim.py`·`sampling.py` 코드 자체는 한 글자도 바꾸지 않고 그대로 동작하게 했다. `requirements-sim.txt`를 직접 `pip install -r`로 설치하게 바꿔 두 파일이 어긋나지 않게 했다.
   - **미확인으로 남긴 것**: `COPY data/recipes.json /app/data/recipes.json`은 원본 그대로 두었다. `recipes.json`은 `sampling.py` 실행 결과물이라 빌드 전에 미리 있어야 하는데, 생성 순서(빌드 전 생성 vs 볼륨 마운트)는 확정하지 않았다 — W1-S1/S4 과제로 남김.

### ruff 린트 예외 처리

seed 원본 5개 파일(`run_sim.py`, `sampling.py`, `sim_batch.py`, `etl.py`, `mgn_smoke_test.py`)은 한 줄에 여러 import·세미콜론 등 스파이크 스타일이라 ruff 기본 규칙(E401/E702/E501/F401 등)에 다수 위반된다. 내용을 고치지 않기로 했으므로, `pyproject.toml`의 `[tool.ruff] extend-exclude`에 이 5개 파일을 넣어 린트 대상에서 뺐다(파일 내용은 손대지 않음). 제외 후 `ruff check .` → `All checks passed!` 재확인.

## 패키지 구조 결정 (문서 공백 보완)

설계서 §6은 `src/sim/`처럼 패키지명 없이 표기하지만, CLAUDE.md·체크리스트(W2-01 `etchpipe/sim/`, W6-05 `python -m etchpipe.retrain`)는 패키지명 `etchpipe`를 전제한다. 이번 작업에서 `src/etchpipe/<component>/` (src 레이아웃)으로 구현해 두 표기를 일치시켰다. `pyproject.toml`(setuptools, `where = ["src"]`)로 `pip install -e .` 가능하도록 구성 — 아래 W1-04 드라이런에서 실제 설치 확인.
