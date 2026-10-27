"""C7: 서빙 최소판(W2-08) — /health, /model, /predict(더미). 실모델 연결은 6주차(W6-08).

API 키 필수: ETCH_API_KEY 환경변수가 없으면 기동 자체를 거부한다(CLAUDE.md: 비밀값은
환경변수로만). /health는 키 없이도 동작(헬스체크는 오케스트레이터가 호출하므로).
"""

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

API_KEY_ENV = "ETCH_API_KEY"


def _require_api_key_env() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} 환경변수가 없어 서빙을 시작할 수 없습니다.")
    return key


_API_KEY = _require_api_key_env()


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


class Recipe(BaseModel):
    ion_flux: float
    etchant_flux: float
    oxygen_flux: float
    ion_mean_energy: float
    ion_exponent: float
    mask_width: float
    mask_height: float


class PredictRequest(BaseModel):
    recipes: list[Recipe]
    times: list[float]


class Profile(BaseModel):
    t: float
    points: list[list[float]]


class PredictResult(BaseModel):
    profiles: list[Profile]
    cd: float
    depth: float
    taper_deg: float
    in_range: bool


class PredictResponse(BaseModel):
    model_version: str
    results: list[PredictResult]


app = FastAPI(title="etchpipe serve")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info(_: None = Depends(check_api_key)) -> dict:
    # 레지스트리 연결은 W6-08. 지금은 더미 champion 고정값.
    return {
        "model_version": "dummy-champion",
        "valid_range": None,
        "gate_result": None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: None = Depends(check_api_key)) -> PredictResponse:
    # 더미: 실제 ViennaPS/모델 연결 없음. 입력 형식만 검증하고 자리만 채운 응답을 준다.
    results = [
        PredictResult(
            profiles=[Profile(t=t, points=[]) for t in req.times],
            cd=0.0,
            depth=0.0,
            taper_deg=0.0,
            in_range=True,
        )
        for _ in req.recipes
    ]
    return PredictResponse(model_version="dummy-champion", results=results)
