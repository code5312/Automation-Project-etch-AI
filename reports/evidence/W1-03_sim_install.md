# W1-03 증거: requirements-sim.txt 클린 설치 확인

- 날짜: 2026-10-27
- 환경: `docker run --rm python:3.13-slim` (호스트가 아닌 컨테이너, 결정 사항: "python은 컨테이너 안에서만 환경 구동")

## 1차 시도 — 실패 (OS 라이브러리 누락)

```
pip install 'ViennaPS==4.5.0' 'ViennaLS==5.8.5' pyvista numpy scipy
python -c "import viennaps, viennals, pyvista, numpy, scipy"
```

설치(pip) 자체는 성공:

```
Successfully installed ViennaLS-5.8.5 ViennaPS-4.5.0 ... numpy-2.5.4 ... pyvista-0.49.1 ... scipy-1.18.1 ... vtk-9.7.1
```

그러나 import는 실패:

```
File "/usr/local/lib/python3.13/site-packages/viennals/__init__.py", line 33, in <module>
    from . import _core as _C  # the binary inside the package
ImportError: libX11.so.6: cannot open shared object file: No such file or directory
```

→ `python:3.13-slim`(Debian slim)에는 VTK(ViennaLS 내부에서 사용)가 필요로 하는 X11 계열 공유 라이브러리가 없다. pip 설치만으로는 재현되지 않으며 OS 패키지가 추가로 필요하다 [확인].

## 2차 시도 — 성공 (apt 패키지 추가)

```
apt-get install -y --no-install-recommends libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6
pip install 'ViennaPS==4.5.0' 'ViennaLS==5.8.5' pyvista numpy scipy
python -c "import viennaps, viennals, pyvista, numpy, scipy; print('OK', viennaps.__version__)"
```

```
--- import smoke (retry with X11 libs) ---
OK 4.5.0
```

## 3차 시도 — pyvista 제외 (seed/ 반영 후 재검증)

seed/의 `Dockerfile.sim`·`run_sim.py`·`sampling.py`·`sim_batch.py`를 실제로 받아 보니 이 C1 스크립트들은 pyvista를 쓰지 않는다(seed의 `Dockerfile.sim`도 `pip install`에 pyvista가 없었다). 설계서 §6: "ViennaPS 환경(시뮬레이션), PhysicsNeMo 환경(**ETL**·학습·평가)"에 따라 `.vtp` 읽기(ETL, `etl.py`)는 train 환경 쪽이다. pyvista를 빼고 재검증:

```
apt-get install -y libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6
pip install 'ViennaPS==4.5.0' 'ViennaLS==5.8.5' numpy scipy
python -c "import viennaps, numpy, scipy; print('OK', viennaps.__version__)"
```

```
OK 4.5.0
pip freeze: numpy==2.5.4 scipy==1.18.1 ViennaLS==5.8.5 ViennaPS==4.5.0
```

numpy/scipy 버전은 pyvista 포함 시와 동일(2.5.4 / 1.18.1) — pyvista 제거가 다른 핀에 영향 없음 [확인].

## 결론

- `requirements-sim.txt`의 최종 핀(ViennaPS==4.5.0, ViennaLS==5.8.5, numpy==2.5.4, scipy==1.18.1 — **pyvista 제외**)은 Python 3.13 컨테이너에서 실제로 설치·import 성공 [확인].
- 추가로 필요한 OS 패키지(`libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6`)를 발견해 `docker/Dockerfile.sim`에 반영함(`RUN apt-get install` 추가) [확인].
- pyvista는 ETL(`etl.py`)이 쓰므로 `requirements-train.txt`로 옮겼다 — `W1-03_train_install.md` 참조.
