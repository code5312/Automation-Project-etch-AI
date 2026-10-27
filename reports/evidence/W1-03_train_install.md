# W1-03 증거: requirements-train.txt 클린 설치 확인

- 날짜: 2026-10-27
- 환경: `docker run --rm python:3.11-slim` + `apt-get install -y build-essential`(gcc/g++, torch_scatter 소스 빌드용)

## 실행

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
pip install nvidia-physicsnemo==2.2.2 torch_geometric==2.8.1
pip install --no-build-isolation torch_scatter==2.1.2
python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import torch_geometric; print('torch_geometric', torch_geometric.__version__)
import torch_scatter; print('torch_scatter OK')
import physicsnemo; print('physicsnemo OK', physicsnemo.__version__)
"
```

## 결과

```
torch 2.14.1+cpu cuda False
torch_geometric 2.8.1
torch_scatter OK
physicsnemo OK 2.2.2
```

```
pip freeze (주요 패키지)
nvidia-physicsnemo==2.2.2
torch==2.14.1+cpu
torch-geometric==2.8.1
torch_scatter==2.1.2
torchvision==0.29.1
```

## 2차 검증 — pyvista/numpy/scipy 추가 (etl.py가 train 환경으로 이동한 뒤)

seed/의 `etl.py`를 받은 뒤 설계서 §6("ETL은 PhysicsNeMo 환경")에 따라 `src/etchpipe/etl/etl.py`로 배치하고, `etl.py`가 쓰는 `pyvista`(.vtp 읽기)·`scipy`(cKDTree)·`numpy`를 이 환경에 추가로 설치해 재검증:

```
pip install pyvista numpy scipy
python -c "import torch, torch_geometric, torch_scatter, physicsnemo, pyvista, numpy, scipy; ..."
```

```
torch 2.14.1+cpu cuda False
physicsnemo 2.2.2
pyvista 0.49.1
numpy 2.4.6
scipy 1.17.1
```

```
pip freeze (주요)
numpy==2.4.6
nvidia-physicsnemo==2.2.2
pyvista==0.49.1
pyvista-validation==0.2.2
scipy==1.17.1
torch==2.14.1+cpu
torch-geometric==2.8.1
torch_scatter==2.1.2
torchvision==0.29.1
vtk==9.7.1
```

numpy(2.4.6)·scipy(1.17.1)는 sim 환경(Python 3.13, numpy==2.5.4/scipy==1.18.1)과 다르다 — train 환경이 Python 3.11이라 pip가 다른 휠을 고른 것이며 두 환경은 독립적이라 문제는 아니다.

## 확인된 사실

- CPU 휠 인덱스(`https://download.pytorch.org/whl/cpu`)를 쓰면 `torch.cuda.is_available()`이 `False`이고 `torch==2.14.1+cpu`로 설치된다 — CUDA 라이브러리(~10GB)를 끌어오지 않는다 [확인]. CLAUDE.md가 "S0에서 확인"이라고 남겨둔 "CPU 휠 인덱스 사용 여부"는 이 결과로 **사용한다**로 답할 수 있다. 다만 이 확인은 로컬 Docker 컨테이너 1회 실행이며, GitHub Actions CI에서의 재현(W1-S0 본래 범위)은 아직 하지 않았다.
- `nvidia-physicsnemo`의 Python import 이름은 `physicsnemo`이다(패키지명과 다름) [확인].
- `torch_scatter==2.1.2`는 `--no-build-isolation`으로 소스에서 빌드되며 `build-essential`(gcc/g++)이 있어야 한다. torch가 먼저 설치되어 있어야 한다(설치 순서 중요) [확인].
- `nvidia-physicsnemo`가 `torchvision==0.29.1`을 함께 끌어온다 [확인].

## 미확인

- GitHub Actions(ubuntu-latest) 러너에서 동일 절차가 재현되는지는 아직 실행하지 않음(W1-S0 범위, 이번 작업 범위 밖) — pr-check 워크플로를 실제로 트리거해 확인해야 한다.
