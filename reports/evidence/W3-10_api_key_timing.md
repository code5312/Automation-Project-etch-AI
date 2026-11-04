# W3-10: 서빙 API 키 비교를 상수시간 비교로 교체

- 날짜: 2026-11-03
- 배경: 코드 리뷰에서 발견 — `check_api_key`가 `x_api_key != _API_KEY`로 비교하는데,
  Python `str.__eq__`는 앞에서부터 다른 문자를 만나면 즉시 멈추는 short-circuit
  비교라 응답시간 차이를 측정해 키를 한 글자씩 추측하는 타이밍 공격이 이론상
  가능함 [제안, 실제 공격 실증은 하지 않음 — 코드 구조상의 알려진 취약 패턴].
- 코드: `src/etchpipe/serve/app.py` (`check_api_key`, `hmac.compare_digest` 사용)
- 테스트: `tests/unit/test_serve.py::test_model_with_wrong_api_key_is_401`(신규)

## 변경 [확인]

```python
def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")
```

`None` 분기를 먼저 둔 이유: `hmac.compare_digest`는 두 인자가 같은 타입(둘 다
`str` 또는 둘 다 `bytes`)이어야 하며 `None`을 받으면 `TypeError`이므로, 헤더가
없는 요청은 비교 전에 걸러낸다.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_serve.py::test_model_with_wrong_api_key_is_401 tests/unit/test_serve.py::test_model_with_api_key_is_200 -q
2 passed
```

틀린 키(`API_KEY + "x"`)는 여전히 401, 맞는 키는 여전히 200 — 비교 함수만
바뀌었고 판정 결과는 동일함을 확인.

## 알려진 환경 이슈(이번 변경과 무관) [확인, pre-existing]

`tests/unit/test_serve.py` 전체를 한 번에 돌리면 Windows + Python 3.14 환경에서
`OSError: [WinError 10014]`(소켓 바인드 실패)가 간헐적으로 발생 — 어느 테스트가
실패하는지도 매 실행마다 바뀜(`test_health_requires_no_api_key`,
`test_predict_without_api_key_is_401`, `test_model_without_api_key_is_401` 등
돌아가며 실패). `check_api_key` 로직과 무관하게 **같은 코드 경로를 쓰는 변경 전
테스트에서도 이미 보고된** 기존 이슈(이전 세션에서 "Windows-환경 소켓 이슈,
이번 세션 작업과 무관"으로 확인된 바로 그 문제) — `TestClient`/`httpx` 전송 계층의
소켓 자원 고갈로 추정, 이번 항목 범위 밖. 개별 테스트 단독 실행 또는 소수 조합
실행으로는 안정적으로 통과함(위 테스트 결과).

## 범위에서 제외한 것

코드 리뷰에서 함께 제안했던 "`_API_KEY`를 모듈 import 시점이 아니라 요청마다
지연 조회"는 이번 항목(hmac 비교 교체)과 별개의 더 큰 변경(기동 실패 동작까지
바뀜)이라 포함하지 않음 — 필요하면 별도 항목으로.
