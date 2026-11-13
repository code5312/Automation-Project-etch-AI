"""C7: 서빙(W2-08 최소판 + W6-08 실모델 연결) — /health, /model, /predict.

API 키 필수: ETCH_API_KEY 환경변수가 없으면 기동 자체를 거부한다(CLAUDE.md: 비밀값은
환경변수로만). /health는 키 없이도 동작(헬스체크는 오케스트레이터가 호출하므로).

**범위 밖 입력은 거부하지 않는다** [확인, DOC-01 2026-11-12]: 체크리스트 W6-08 원문은
"2배 초과 422"였지만 `project-definition.md` §4·`project-design.md` §C7 둘 다 "유효
범위 밖은 거부하지 않고 `in_range=false`로 경고만 한다"를 명시한다 — 우선순위
규칙(definition > design > checklist)에 따라 상위 문서를 따른다(사용자 결정).

**오류 응답 정제(W6-09)**: 예상 못 한 예외는 경로·스택을 노출하지 않는다 — `/predict`는
route 안에서 직접 잡아 422로 바꾸고(1차 방어, "이상 입력 퍼징에서 5xx 없음" 요건과
직결), 그 외 경로는 `@app.exception_handler(Exception)`이 2차 방어막으로 잡는다. 두
미들웨어(`_BodySizeLimitMiddleware`/`_RateLimitMiddleware`)를 순수 ASGI 방식으로 쓴
이유도 이 핸들러가 실제로 불리게 하려는 것(`BaseHTTPMiddleware`는 예외 전파를
막는 Starlette의 알려진 함정이 있음, 실측 확인).
"""

import hmac
import json
import logging
import math
import os
import pathlib
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from etchpipe.registry.registry import Registry
from etchpipe.serve.inference import ChampionUnavailable, load_champion, predict_recipe

logger = logging.getLogger("etchpipe.serve")

API_KEY_ENV = "ETCH_API_KEY"
MAX_BATCH_RECIPES = 20  # [제안] 한 요청의 레시피 배치 크기 상한
MAX_TIMES = 20  # [제안] 레시피당 요청 시간 개수 상한
MAX_BODY_BYTES = 1_000_000  # [제안] 1MB — 요청 크기 제한
RATE_LIMIT_PER_MINUTE = 60  # [제안] API 키당 분당 요청 수 상한(단일 프로세스 기준)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DASHBOARD_STATIC_DIR = REPO_ROOT / "src" / "etchpipe" / "dashboard" / "static"
MODEL_COMPARISON_PATH = REPO_ROOT / "reports" / "eval" / "W6-12_model_comparison.json"
REGISTRY_ROOT = REPO_ROOT / "registry"


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
    # gt=0/le=...: 물리량이 음수·0일 수 없다는 상식적 하한 + 상한 — 레지스트리
    # valid_range(학습 범위) 경고와는 별개(저건 거부 안 함, 이건 애초에 말이 안 되는 값
    # 거부)[제안]. 상한은 sampling.py SPACE(학습 범위)의 10배 — OOD 탐색은 넉넉히
    # 허용하면서도 물리적으로 터무니없는 값은 막는다. **mask_width/mask_height는 훨씬
    # 빡빡하게(≤25)** — `inference.build_initial_graph`의 `ps.Domain(xExtent=30, ...)`에
    # 직접 들어가는 기하 치수라, 아주 큰 값(실측: 1e9)을 주면 ViennaPS 메시 생성이
    # 멈추거나 응답 없이 죽는다(2026-11-12 실측 확인 — W6-09 작업 중 발견).
    # `allow_inf_nan=False`: gt/le만으로는 NaN/Infinity를 못 걸러낸다(둘 다 비교 결과가
    # 항상 False라 pydantic-core가 통과시킴, 실측 확인 — NaN/Infinity가 실제 추론까지
    # 흘러가서 출력이 NaN이 되고, FastAPI가 그 응답을 JSON으로 직렬화하는 단계에서
    # `ValueError: Out of range float values are not JSON compliant`로 터져 500이 됨,
    # 이 과정은 `predict()`의 try/except 바깥(응답 직렬화 시점)이라 422로 못 바꿈 —
    # W6-09 100개 퍼징 테스트로 발견).
    ion_flux: float = Field(gt=0, le=300.0, allow_inf_nan=False)
    etchant_flux: float = Field(gt=0, le=135_000.0, allow_inf_nan=False)
    oxygen_flux: float = Field(gt=0, le=24_000.0, allow_inf_nan=False)
    ion_mean_energy: float = Field(gt=0, le=2_000.0, allow_inf_nan=False)
    ion_exponent: float = Field(gt=0, le=10_000.0, allow_inf_nan=False)
    mask_width: float = Field(gt=0, le=25.0, allow_inf_nan=False)
    mask_height: float = Field(gt=0, le=25.0, allow_inf_nan=False)


class PredictRequest(BaseModel):
    recipes: list[Recipe] = Field(min_length=1, max_length=MAX_BATCH_RECIPES)
    times: list[float] = Field(min_length=1, max_length=MAX_TIMES)

    @field_validator("times")
    @classmethod
    def _times_positive_and_finite(cls, v: list[float]) -> list[float]:
        # math.isfinite(nan/inf) == False — `t <= 0`만으로는 NaN을 못 걸러낸다(Recipe의
        # allow_inf_nan=False와 같은 이유, W6-09 퍼징으로 발견).
        if any(not math.isfinite(t) or t <= 0 for t in v):
            raise ValueError("times는 모두 양수이고 유한해야 합니다")
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


class _BodySizeLimitMiddleware:
    """요청 크기 제한 — `Content-Length`가 상한을 넘으면 본문을 읽지 않고 바로 413.

    **순수 ASGI 미들웨어**(`BaseHTTPMiddleware` 상속 아님)[확인, 2026-11-12 실측] —
    `BaseHTTPMiddleware`는 route 안에서 던진 예외가 `@app.exception_handler`를 안 거치고
    그대로 새는 Starlette의 알려진 함정이 있다(실측: `/model`에서 `ChampionUnavailable`이
    아닌 예외를 테스트로 던져보니 핸들러가 전혀 안 불림). W6-09 "경로·스택 미노출"
    요건과 직결돼서 순수 ASGI 방식으로 바꿨다."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            await _json_response(413, {"detail": "request body too large"})(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _RateLimitMiddleware:
    """API 키당 분당 요청 수 제한 [제안, 단일 프로세스 메모리 기준 — 여러 워커·분산
    배포에서는 각 프로세스가 독립적으로 셈. 서빙 최적화는 이 프로젝트 목적이 아니라는
    설계서 C7 방향과 같은 수준의 최소 구현]. `_BodySizeLimitMiddleware`와 같은 이유로
    순수 ASGI 방식."""

    def __init__(self, app, limit_per_minute: int = RATE_LIMIT_PER_MINUTE) -> None:
        self.app = app
        self.limit = limit_per_minute
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_start_min, count)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        raw_key = headers.get(b"x-api-key")
        if raw_key is not None:
            key = raw_key.decode("latin-1")
        else:
            client = scope.get("client")
            key = client[0] if client else "unknown"
        window = int(time.time() // 60)
        window_start, count = self._counters.get(key, (window, 0))
        if window_start != window:
            window_start, count = window, 0
        count += 1
        self._counters[key] = (window_start, count)
        if count > self.limit:
            await _json_response(429, {"detail": "rate limit exceeded"})(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _json_response(status_code: int, content: dict) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=content)


app = FastAPI(title="etchpipe serve")
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_BodySizeLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """W6-09: FastAPI 기본 422 핸들러는 거부된 원본 입력값을 에러 응답에 그대로
    echo하는데, 그 값이 NaN/Infinity면 **에러 응답 자체**가 JSON 직렬화에 실패해
    500이 된다(실측 확인, 100개 퍼징 테스트로 발견). 원본 값(`"input"` 키)은 빼고
    필드 경로·메시지·종류만 돌려준다 — 항상 안전하게 직렬화된다."""
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _json_response(422, {"detail": errors})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """W6-09: 처리 안 한 예외가 경로·스택을 그대로 노출하지 않게 한다(FastAPI 기본값도
    debug=False면 안 노출하지만, 명시적으로 고정해 둔다) — 실제 예외는 서버 로그에만
    남긴다. 라우트 안에서 미리 잡아서 422로 바꾸는 쪽(아래 `predict`)이 1차 방어고,
    이건 그걸 놓친 경우의 2차 방어막[제안]."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return _json_response(500, {"detail": "internal error"})


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


def _run_predict(req: PredictRequest) -> PredictResponse:
    try:
        raw_results = [predict_recipe(recipe.model_dump(), req.times) for recipe in req.recipes]
    except ChampionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        # W6-09: Field(gt=0/le=...)로 걸러지지 않는, 추론 단계에서만 드러나는 비정상
        # 입력(예: 기하가 깨지는 조합)은 422로 돌린다 — 5xx를 아예 안 낸다. 실제 예외는
        # 서버 로그에만(경로·스택 미노출).
        logger.exception("predict failed for recipe batch")
        raise HTTPException(status_code=422, detail="요청을 처리할 수 없습니다") from e
    # model_version은 응답 최상위에만 둔다(설계서 §7) — predict_recipe 각 호출이 같은
    # champion에서 나온 값이라 첫 결과에서 한 번만 꺼낸다.
    model_version = raw_results[0]["model_version"]
    results = [
        PredictResult(**{k: v for k, v in r.items() if k != "model_version"})
        for r in raw_results
    ]
    return PredictResponse(model_version=model_version, results=results)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: None = Depends(check_api_key)) -> PredictResponse:
    return _run_predict(req)


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest) -> PredictResponse:
    """W7-01 대시보드용 공개 엔드포인트 — 같은 출처라 브라우저는 API 키를 몰라도 된다.
    서버 프로세스 안에서 `/predict`와 같은 로직(`_run_predict`)을 그대로 호출할 뿐,
    키를 "대신 붙여서 전달"하는 별도 HTTP 홉은 없다(같은 프로세스라 불필요) — 외부에
    노출할 때 브라우저 쪽 JS에 `ETCH_API_KEY`가 그대로 보이는 걸 막는 게 목적이라,
    전달 경로가 몇 홉인지는 중요하지 않다[확인]. 요청 크기·속도 제한(`_RateLimitMiddleware`)
    은 이 경로에도 그대로 적용된다(클라이언트 IP 기준, 60/분)."""
    return _run_predict(req)


@app.get("/api/models/compare")
def api_models_compare() -> dict:
    """W7-01 대시보드 화면 ②(모델 비교표) — `scripts/run_model_comparison.py`(W6-12)가
    만든 `reports/eval/W6-12_model_comparison.json`을 그대로 공개. 사전 계산된 정적
    결과라 요청마다 재계산하지 않는다[확인]."""
    if not MODEL_COMPARISON_PATH.exists():
        raise HTTPException(status_code=503, detail="모델 비교 결과가 아직 없습니다")
    return json.loads(MODEL_COMPARISON_PATH.read_text())


@app.get("/api/gate/history")
def api_gate_history() -> dict:
    """W7-01 대시보드 화면 ③(게이트 이력) — `Registry.history()`(승격/거부 이력)를
    그대로 공개. `registry/registry.py`가 이미 테스트로 검증한 파싱 로직을 그대로
    재사용한다(직접 `history.jsonl`을 다시 파싱하지 않음, [확인])."""
    return {"entries": Registry(REGISTRY_ROOT).history()}


# W7-01: 대시보드 정적 파일을 같은 FastAPI 앱·같은 포트에서 서빙 — 별도 서버·CORS 설정
# 없이 "같은 출처의 /api/predict" 전제(dashboard/README.md)를 그대로 만족시킨다. 명시적
# 라우트(health/model/predict/api/*)를 전부 등록한 뒤 마지막에 마운트해야 경로가 겹치지
# 않는다[확인] — StaticFiles(html=True)는 "/"를 index.html로, 못 찾는 경로는 404로 돌린다.
if DASHBOARD_STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_STATIC_DIR), html=True), name="dashboard")
