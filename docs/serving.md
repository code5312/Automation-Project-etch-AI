# 서빙 로컬 실행 (W6-10)

사전 조건: `scripts/register_champion.py`로 `registry/`를 먼저 채워 둬야 한다
(`registry/`는 `.gitignore` 대상 — 로컬에만 존재, W6-08 참고). 이미지는
`docker build -f docker/Dockerfile.serve -t etch-serve:0.1 .`로 미리 빌드.

## 방식 ① docker-compose

```
ETCH_API_KEY=<키> docker compose up -d
curl http://localhost:8000/health
docker compose down
```

`docker-compose.yml`이 `src/`·`registry/`·`runs/mgn_tuned/`를 읽기 전용으로 마운트하고
8000번 포트로 띄운다.

## 방식 ② 직접 실행(compose 없이 `docker run`)

```
docker run -d --name etch-serve-direct -p 8001:8000 \
  -e ETCH_API_KEY=<키> \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/registry:/app/registry:ro" \
  -v "$(pwd)/runs/mgn_tuned:/app/runs/mgn_tuned:ro" \
  -w /app etch-serve:0.1 \
  uvicorn etchpipe.serve.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8001/health
docker stop etch-serve-direct && docker rm etch-serve-direct
```

같은 이미지·같은 볼륨을 compose 없이 그대로 쓴다 — compose는 그 커맨드를 YAML로
대신 적어 둔 것뿐이라, 두 방식의 실행 결과는 동일해야 한다.

## 결과 [확인, 2026-11-12 실측]

| | `/health` | `/predict` 지연(3회, recipe 1개·times 2개) |
|---|---|---|
| ① docker-compose | 200 OK | 18.967s / 19.418s / 20.059s |
| ② 직접 실행(docker run) | 200 OK | 30.436s / 19.161s / 19.923s |

**조건**: CPU만(GPU 없음, `Warp CUDA warning` 로그 확인), champion=`mgn_tuned_ep2`,
recipe 1개·`times=[2.0, 5.0]`(2스텝 롤아웃), `fix_mask`/`guard` 켬. 첫 호출(②의
30.4s)이 나머지(~19~20s)보다 느린 건 모듈 지연 임포트(physicsnemo 등)·torch JIT
웜업으로 보인다 — 방식 자체의 차이가 아니라 **프로세스 기동 후 첫 요청의 1회성
비용**[제안, 반복 실측 전까지는 추정]. 두 방식 모두 안정화 후 지연은 거의 동일
(~19~20s) — 이 프로젝트가 서빙 성능 최적화를 목적으로 하지 않는다는 설계서 C7
방향과 일치[확인, 실측].
