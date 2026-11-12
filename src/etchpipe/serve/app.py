"""C7: 서빙(W2-08 최소판 + W6-08 실모델 연결) — /health, /model, /predict.

API 키 필수: ETCH_API_KEY 환경변수가 없으면 기동 자체를 거부한다(CLAUDE.md: 비밀값은
환경변수로만). /health는 키 없이도 동작(헬스체크는 오케스트레이터가 호출하므로).

**범위 밖 입력은 거부하지 않는다** [확인, DOC-01 2026-11-12]: 체크리스트 W6-08 원문은
"2배 초과 422"였지만 `project-definition.md` §4·`project-design.md` §C7 둘 다 "유효
범위 밖은 거부하지 않고 `in_range=false`로 경고만 한다"를 명시한다 — 우선순위
규칙(definition > design > checklist)에 따라 상위 문서를 따른다(사용자 결정).
"""

import hmac
import os
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from etchpipe.serve.inference import ChampionUnavailable, load_champion, predict_recipe

API_KEY_ENV = "ETCH_API_KEY"
MAX_BATCH_RECIPES = 20  # [제안] 한 요청의 레시피 배치 크기 상한
MAX_TIMES = 20  # [제안] 레시피당 요청 시간 개수 상한
MAX_BODY_BYTES = 1_000_000  # [제안] 1MB — 요청 크기 제한
RATE_LIMIT_PER_MINUTE = 60  # [제안] API 키당 분당 요청 수 상한(단일 프로세스 기준)


def _require_api_key_env() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} 환경변수가 없어 서빙을 시작할 수 없습니다.")
    return key


_API_KEY = _require_api_key_env()


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # W3-10: 상수시간 비교(hmac.compare_digest) — `!=`는 앞에서부터 다른 문자를
    # 만나면 바로 멈추는 short-circuit 비교라 응답시간차로 키를 추측하는 타이밍
    # 공격 이론상 가능 [제안, 코드리뷰로 발견].
    if x_api_key is None or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


class Recipe(BaseModel):
    # gt=0: 물리량이 음수·0일 수 없다는 상식적 하한 — 레지스트리 valid_range(학습 범위)
    # 경고와는 별개(저건 거부 안 함, 이건 애초에 말이 안 되는 값 거부) [제안].
    ion_flux: float = Field(gt=0)
    etchant_flux: float = Field(gt=0)
    oxygen_flux: float = Field(gt=0)
    ion_mean_energy: float = Field(gt=0)
    ion_exponent: float = Field(gt=0)
    mask_width: float = Field(gt=0)
    mask_height: float = Field(gt=0)


class PredictRequest(BaseModel):
    recipes: list[Recipe] = Field(min_length=1, max_length=MAX_BATCH_RECIPES)
    times: list[float] = Field(min_length=1, max_length=MAX_TIMES)

    @field_validator("times")
    @classmethod
    def _times_positive(cls, v: list[float]) -> list[float]:
        if any(t <= 0 for t in v):
            raise ValueError("times는 모두 양수여야 합니다")
        return v


class Profile(BaseModel):
    t: float
    points: list[list[float]]


class PredictResult(BaseModel):
    profiles: list[Profile]
    cd: float
    depth: float
    taper_deg: float
    in_range: bool
    clamped: bool = False


class PredictResponse(BaseModel):
    model_version: str
    results: list[PredictResult]


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """요청 크기 제한 — `Content-Length`가 상한을 넘으면 본문을 읽지 않고 바로 413."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_BODY_BYTES:
            return _json_response(413, {"detail": "request body too large"})
        return await call_next(request)


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """API 키당 분당 요청 수 제한 [제안, 단일 프로세스 메모리 기준 — 여러 워커·분산
    배포에서는 각 프로세스가 독립적으로 셈. 서빙 최적화는 이 프로젝트 목적이 아니라는
    설계서 C7 방향과 같은 수준의 최소 구현]."""

    def __init__(self, app, limit_per_minute: int = RATE_LIMIT_PER_MINUTE) -> None:
        super().__init__(app)
        self.limit = limit_per_minute
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_start_min, count)

    async def dispatch(self, request: Request, call_next):
        key = request.headers.get("x-api-key")
        if not key:
            key = request.client.host if request.client else "unknown"
        window = int(time.time() // 60)
        window_start, count = self._counters.get(key, (window, 0))
        if window_start != window:
            window_start, count = window, 0
        count += 1
        self._counters[key] = (window_start, count)
        if count > self.limit:
            return _json_response(429, {"detail": "rate limit exceeded"})
        return await call_next(request)


def _json_response(status_code: int, content: dict) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=content)


app = FastAPI(title="etchpipe serve")
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_BodySizeLimitMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info(_: None = Depends(check_api_key)) -> dict:
    try:
        champion = load_champion()
    except ChampionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    card = champion["card"]
    return {
        "model_version": champion["model_id"],
        "valid_range": card["valid_range"],
        "gate_result": card["gate_result"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: None = Depends(check_api_key)) -> PredictResponse:
    try:
        raw_results = [predict_recipe(recipe.model_dump(), req.times) for recipe in req.recipes]
    except ChampionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    # model_version은 응답 최상위에만 둔다(설계서 §7) — predict_recipe 각 호출이 같은
    # champion에서 나온 값이라 첫 결과에서 한 번만 꺼낸다.
    model_version = raw_results[0]["model_version"]
    results = [
        PredictResult(**{k: v for k, v in r.items() if k != "model_version"})
        for r in raw_results
    ]
    return PredictResponse(model_version=model_version, results=results)
