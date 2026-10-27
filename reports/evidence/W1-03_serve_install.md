# W1-03 증거: requirements-serve.txt 클린 설치 확인

- 날짜: 2026-10-27
- 환경: `docker run --rm python:3.11-slim`

## 실행

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install fastapi==0.162.0 uvicorn==0.54.0 pydantic==2.14.1 numpy==2.4.6
python -c "
import torch, fastapi, uvicorn, pydantic, numpy
print('torch', torch.__version__)
print('fastapi', fastapi.__version__)
print('uvicorn', uvicorn.__version__)
print('pydantic', pydantic.VERSION)
print('numpy', numpy.__version__)
"
```

## 결과

```
torch 2.14.1+cpu
fastapi 0.162.0
uvicorn 0.54.0
pydantic 2.14.1
numpy 2.4.6
```

전체 설치(`pip freeze`)가 CUDA 패키지를 포함하지 않음을 확인(torch==2.14.1+cpu만 존재) [확인].

## 비고

- `requirements-sim.txt`의 numpy(2.5.4)와 `requirements-serve.txt`의 numpy(2.4.6) 버전이 다르다. 두 환경은 분리되어 있어(CLAUDE.md "환경 3종 분리") 문제는 아니지만, 각 환경에서 핀을 독립적으로 재현했다는 뜻이다 — 의도적으로 통일하지 않았다.
