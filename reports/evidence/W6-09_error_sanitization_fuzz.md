# W6-09: 오류 응답 정제 + 이상 입력 100개 퍼징

- 날짜: 2026-11-12
- 코드: `src/etchpipe/serve/app.py`(수치 상·하한, NaN/Infinity 거부, 예외 핸들러 2종,
  순수 ASGI 미들웨어로 교체), `scripts/fuzz_predict.py`(신설)

## 실제로 찾은 버그 3종 [확인, 실측]

1. **극단값 geometry 멈춤**: `mask_width=1e9`를 주면 ViennaPS 메시 생성이 멈추거나
   응답 없이 죽는다(`ps.Domain(xExtent=30, ...)`에 직접 들어가는 값이라). →
   `mask_width`/`mask_height`에 `le=25.0` 상한 추가(기존엔 `gt=0`만 있었음).
2. **NaN/Infinity가 gt/le를 통과**: `Field(gt=0, le=...)`만으로는 NaN/Infinity를
   못 걸러낸다(비교 결과가 항상 `False`라 pydantic-core가 통과시킴) — 실제 추론까지
   흘러가 출력이 NaN이 되고, 응답 JSON 직렬화 단계(`predict()`의 try/except 바깥)에서
   `ValueError: Out of range float values are not JSON compliant`로 500. →
   `allow_inf_nan=False` 추가(Recipe 7개 필드 + `times` 커스텀 검증).
3. **422 에러 응답 자체가 깨짐**: FastAPI 기본 검증 에러 핸들러가 거부된 원본 값을
   에러 응답에 그대로 echo하는데, 그 값이 NaN/Infinity면 **에러 응답 자체**가
   직렬화 실패로 500이 된다 — 버그 2를 고친 뒤에도 재현됨. →
   `RequestValidationError` 전용 핸들러로 교체, 원본 입력값(`"input"` 키) 제외하고
   `loc`/`msg`/`type`만 반환.

## 부수 발견: `BaseHTTPMiddleware`가 예외 핸들러를 막음 [확인, 실측]

`_BodySizeLimitMiddleware`/`_RateLimitMiddleware`가 `BaseHTTPMiddleware`를 상속했을 때,
route 안에서 안 잡은 예외가 `@app.exception_handler(Exception)`을 전혀 안 거치고 그대로
샜다(Starlette의 알려진 함정). 순수 ASGI 미들웨어(`__call__(scope, receive, send)`)로
바꿔서 해결.

## 결과 [확인] — 100개 퍼징, 5xx 0건

```
PYTHONPATH=src python scripts/fuzz_predict.py   (etch-serve:0.1, 실제 champion)
→ total=100 5xx_count=0
```

7개 필드 × {음수, 0, 상한초과(1e12), NaN, Infinity} = 35종 + 마스크 경계값 2종 +
times 이상값 5종 + 전체 필드 동시 이상값 2종 + 나머지는 정상 변주로 100개를 채움.
전체 리포트: `build/fuzz_predict_report.json`(gitignore 대상, 재실행 때마다 갱신).

## 테스트

```
python -m ruff check src/etchpipe/serve/ scripts/fuzz_predict.py tests/unit/test_serve.py
  → All checks passed
PYTHONPATH=src python -m pytest tests/unit/test_serve.py -v (etch-serve:0.1 또는
  fastapi/httpx 추가 설치한 etch-train:0.1) → 26 passed
```

`scripts/fuzz_predict.py`는 실제 champion(레지스트리 등록 필요)이 있어야 돌아가서
CI에는 안 걸었다(`run_sim.py`/`run_drift_sim.py`와 같은 이유 — 로컬 실측 전용).
