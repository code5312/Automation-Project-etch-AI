# W1-S4: Docker/kind 경로 판정

- 목표: Docker 설치 확인 → `Dockerfile.sim` 빌드 → kind 클러스터 → `sim-job-local.yaml` 파드 4개 `Completed` → 파드 1개 삭제 후 재시도·이어하기 확인. **docker+kind 경로로 판정하고 이유를 기록.**
- 시간 상자: 1시간(실제로는 중간에 두 가지 실제 문제를 만나 총 약 1시간 40분 소요 — 아래에 그대로 기록. "시간 상자 초과 시 즉시 중단" 규칙은 1차 시도에서 실제로 적용했다)
- 사용자 결정: 기존 kind 클러스터(`etch-pipeline-control-plane`, 3시간 넘게 떠 있던 것) 삭제 후 처음부터 재구성.

## 0) 준비

```
./kind.exe delete cluster --name etch-pipeline
# data/recipes.json 생성 (sampling.make_recipes(20, seed=20261027))
docker build -f docker/Dockerfile.sim -t etch-sim:0.1 .
```
이미지 빌드 성공(25.9초, ViennaPS==4.5.0/ViennaLS==5.8.5/numpy/scipy 설치 + apt X11 라이브러리 포함).

## 1) 실제 문제 #1: Docker Desktop 포트 퍼블리시 전체 장애 (환경 문제, kind와 무관)

`kind create cluster`가 매번 다른 임의 포트로 똑같이 실패:
```
docker: Error response from daemon: ports are not available: ... listen: The system detected an invalid pointer address ...
```
평범한 `nginx:alpine`을 `-p`로 띄워도 같은 에러가 나서, **kind 문제가 아니라 Docker Desktop 자체의 포트 퍼블리시 기능이 깨진 상태**임을 확인했다. 사용자가 Docker Desktop을 재시작(버전도 28.5.1→29.9.2로 올라감)한 뒤 재확인하니 포트 퍼블리시가 정상 동작했다. **이 문제는 docker+kind 경로 자체의 결함이 아니라 이번 세션의 Docker Desktop 상태 문제였다** — 재현성 평가에는 "클린 환경에서 이 문제가 또 발생할 수 있다"는 리스크로만 반영.

## 2) 실제 문제 #2: cpu 리소스 과소 할당 → 1차 시도 시간 상자 초과

`sim-job-local.yaml` 원본의 `resources.limits.cpu: "1"`로 Job을 돌렸더니 **30분 동안 파드 2개가 레시피 1건도 못 끝냄**(`Process time` 로그가 0.65~0.96/1.0 min 구간에서 맴돔). W1-S1에서 이미 확인한 "ViennaPS 레이트레이싱이 내부적으로 약 12~13 스레드를 쓴다"는 사실과 맞물려, cpu 1코어로 묶이면 10배 이상 느려진다. **시간 상자(1시간) 초과가 확실해 즉시 Job을 중단**하고(`kubectl delete job`), `sim-job-local.yaml`의 cpu를 `"1"→"6"`(parallelism=2 기준 2×6=12코어, 호스트 14코어 내)으로 올려 재시도. `sim-job.yaml`(실제 클러스터용, parallelism=8 cpu=2)도 같은 문제가 있을 수 있다는 주석을 남겼다(실제 노드 코어 수를 모르므로 숫자는 바꾸지 않음, [미확인]).

## 3) 2차 시도: 4개 파드 Completed

cpu=6으로 재적용 후 **24분 만에 4/4 Complete**. 레시피별 시간(K=2, kind 환경):

| run_id | kind(cpu=6) | S1(제한 없음) | 배율 |
|---|---|---|---|
| r0001 | 153.4s | 77.2s | 2.0x |
| r0002 | 78.8s | 48.2s | 1.6x |
| r0003 | 235.2s | 110.9s | 2.1x |
| r0005 | 18.6s | 10.7s | 1.7x |
| r0006 | 193.3s | 96.2s | 2.0x |
| r0007 | 106.4s | 42.0s | 2.5x |
| r0008 | 209.6s | 91.7s | 2.3x |
| r0009 | 90.1s | 46.4s | 1.9x |
| r0010 | 121.1s | 47.2s | 2.6x |
| r0011 | 318.6s | 133.1s | 2.4x |
| r0012 | 366.1s | 159.2s | 2.3x |
| r0013 | 61.2s | 31.9s | 1.9x |
| r0014 | 93.1s | 40.4s | 2.3x |
| r0015 | 118.6s | 56.3s | 2.1x |
| r0016 | 387.4s | 110.2s | 3.5x |
| r0017 | 50.4s | 15.9s | 3.2x |
| r0018 | 45.6s | 16.3s | 2.8x |
| r0019 | 142.1s | 51.9s | 2.7x |

평균 약 2.3배 느림(cpu=1일 때 10배 이상이던 것에서 크게 개선됐지만, ViennaPS가 원하는 ~12~13스레드에 아직 못 미침). **3주차 본 데이터 생성(W3-02)에서는 cpu를 12 근처로 올리는 쪽이 처리량에 유리할 것으로 보인다 [제안].**

## 4) 재시도·이어하기 확인

PVC를 비우고(`rm -rf /data/raw`) 깨끗하게 재실행. 샤드 0 파드가 `r0000`, `r0004`를 끝낸 뒤(진행 중) **파드를 강제로 삭제**:

```
kubectl delete pod etch-sim-local-0-692fj
```

즉시 Job 컨트롤러가 같은 인덱스(0)로 새 파드(`etch-sim-local-0-fr6jm`)를 자동 생성. 새 파드 로그:
```
shard 0/4: 5 recipes, K=2
skip r0000
skip r0004
ok r0008 209.6s
ok r0012 366.1s
ok r0016 387.4s
```
**확인됨**: 죽은 파드가 이미 끝낸 레시피는 정확히 skip하고 나머지부터 이어간다(PVC가 공유 상태 저장소 역할을 제대로 함). 이는 "재시도"(Job이 파드를 다시 만듦)와 "이어하기"(`sim_batch.py`의 `meta.json` 존재 확인 skip 로직) 둘 다 실제로 동작함을 보여준다.

## 판정

**docker+kind 경로 확정**(xargs -P 대체 경로로 넘어가지 않음). 이유:
1. 이미지 빌드, 클러스터 생성, PVC, Indexed Job 샤딩이 모두 실제로 동작했다.
2. 파드 삭제 → 자동 재생성 → `meta.json` 기준 skip → 이어하기까지 실측으로 확인됐다(리스크 표의 "로컬 kind 1회 성공" 목표 달성).
3. 만난 두 문제(Docker Desktop 포트 버그, cpu 과소 할당)는 둘 다 **docker+kind 경로 자체의 결함이 아니라 이번 환경 설정 문제**였고 실제로 고쳤다.

## 대안

불필요(docker+kind 성공). 다만 아래는 다음 단계로 넘긴다:
- `sim-job.yaml`(실클러스터용) 의 cpu/parallelism은 실제 노드 사양을 알아야 재조정 가능 [미확인] — W3-02에서 확정.
- Docker Desktop 포트 퍼블리시 장애가 다시 발생할 경우의 대응(재시작)은 리포지토리 문서가 아니라 로컬 운영 이슈로 분류, README에는 올리지 않음.
