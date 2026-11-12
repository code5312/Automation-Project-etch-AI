# W6-10: `docker-compose.yml` + 로컬 직접 실행

- 날짜: 2026-11-12
- 코드: `docker-compose.yml`(신설), `docs/serving.md`(신설)

## 실행 [확인, 실측]

두 방식 모두 같은 이미지(`etch-serve:0.1`)·같은 볼륨(`src/`·`registry/`·
`runs/mgn_tuned/`, 전부 읽기 전용)으로 실제로 띄워 `/health`·`/predict`를 확인했다.

```
# ① docker-compose
ETCH_API_KEY=test-key docker compose up -d
curl http://localhost:8000/health        -> {"status":"ok"} 200
curl http://localhost:8000/model (x-api-key: test-key) -> 200, 실제 레지스트리 정보
docker compose down

# ② 직접 실행(docker run, compose 없이)
docker run -d --name etch-serve-direct -p 8001:8000 -e ETCH_API_KEY=test-key \
  -v ...src:/app/src:ro -v ...registry:/app/registry:ro -v ...runs/mgn_tuned:/app/runs/mgn_tuned:ro \
  -w /app etch-serve:0.1 uvicorn etchpipe.serve.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8001/health        -> {"status":"ok"} 200
```

API 키 검증도 실제로 동작함을 확인(`x-api-key` 틀리거나 빈 값 -> 401, 정답 -> 200).

## 지연 측정 [확인, 실측, 조건 병기]

| | `/predict` 지연(3회, recipe 1개·times=[2.0,5.0]) |
|---|---|
| ① docker-compose | 18.967s / 19.418s / 20.059s |
| ② 직접 실행 | 30.436s / 19.161s / 19.923s |

조건: CPU만(GPU 없음), champion=`mgn_tuned_ep2`, `fix_mask`/`guard` 켬. ②의 첫
호출이 느린 건(30.4s) 프로세스 기동 후 첫 요청의 1회성 비용(지연 임포트·torch JIT
웜업)으로 보임 — 방식 자체의 차이는 아님[제안]. 안정화 후에는 두 방식이 거의
동일(~19~20s). 상세는 `docs/serving.md`.

## 알려진 한계

- `docker-compose.yml`은 레지스트리 재등록(W6-04 루프가 실제 새 champion을
  승격하면)이나 복수 champion 지원은 안 다룬다 — 지금은 `runs/mgn_tuned/`만
  고정 마운트.
