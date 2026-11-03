# W3-01: probe 8건 depth_max 적용 kind Job 재측정 (사후 확인)

- 날짜 확인: 2026-11-03 (측정 자체는 PVC 타임스탬프 기준 2026-10-28 실행된 것으로 추정, 이번 세션에서 **사후에 발견·검증**)
- 목적: `W3-01_depth_max_revalidation.md`가 권고한 "probe 8건 전체를 재빌드 이미지로 depth_max 적용 재측정"을 수행하려 했으나, `deploy/k8s/sim-job-probe.yaml`로 kind에 이미 Job이 적용되어 **8/8 완주된 상태**로 PVC(`etch-data`)에 남아 있는 것을 발견함. 원 실행 명령은 기록이 없어 [미확인], 결과 데이터는 이번에 직접 조회해 [확인]함.

## 확인 방법

```
# busybox + etch-data PVC 마운트 임시 파드(스크래치 디렉터리에만 작성, 저장소에는 커밋 안 함)
kubectl --context kind-etch-pipeline apply -f <scratch>/pvc-inspect.yaml
kubectl --context kind-etch-pipeline logs pvc-inspect   # /data/raw/*/meta.json 전체 출력 확인
kubectl --context kind-etch-pipeline delete pod pvc-inspect
```

출력된 `meta.json` 8건 모두 `depth_max: 50.0` 필드를 포함하고 있어, 이미지 재빌드 버그(`W3-01_probe.md`) 이후 실행된 것이 확인됨.

## 측정값 (K=20 상한, depth_max=50, rays=300, grid_delta=0.3, cpu=6 제한, kind Indexed Job parallelism=2)

| run_id | ion_flux | actual_k | stopped_early | final_depth(µm) | wall_sec |
|---|---|---|---|---|---|
| r0005 | 3.34 | 20/20 | false | 4.8 | 240.2 |
| r0002 | 5.25 | 20/20 | false | 31.2 | 1026.2 |
| r0001 | 6.45 | 18/20 | true | 51.3 | 1424.8 |
| r0006 | 7.17 | 13/20 | true | 52.2 | 1408.1 |
| r0007 | 9.89 | 20/20 | false | 33.3 | 810.3 |
| r0000 | 14.79 | 15/20 | true | 52.8 | 1382.4 |
| r0003 | 21.39 | 13/20 | true | 50.4 | 1262.7 |
| r0004 | 25.90 | 5/20 | true | 55.5 | 1530.8 |

- 평균 wall_sec: **1135.7초(약 18.9분)**, 최소 240.2 / 최대 1530.8 (비율 6.4배) — K=2 표본(W1-S1)의 23.6배보다 훨씬 좁음. **depth_max 도입으로 긴 꼬리가 줄어든 것이 실측으로 확인됨.**
- depth_max 조기종료 발생: 8건 중 5건(ion_flux 6.45 이상 거의 전부). 안 걸린 3건(r0005, r0002, r0007)은 낮거나 중간 ion_flux로 20스텝 내 50µm에 못 미침.
- 주의: r0004는 `W3-01_depth_max_revalidation.md`의 단독 `docker run`(cpu 제한 없음) 결과(386.2초)와 actual_k=5로 스텝 수는 일치하지만 wall_sec은 1530.8초로 약 4배 더 걸림 — **cpu=6 제한 효과**(물리 결과는 동일, 속도만 차이). 두 측정 모두 depth_max가 올바르게 작동함을 보여주는 것은 동일.

## N=410(train300+val40+test40+ood20+golden10) 시간 추정

평균 1135.7초/건, kind parallelism=2(cpu=6×2=12/14 vCPU) 기준:

| N | parallelism=2 | parallelism=4 |
|---|---|---|
| 410 | 약 64.7시간(2.7일) | 약 32.3시간(1.35일) |
| 400(폴백) | 약 63.1시간(2.6일) | 약 31.5시간(1.3일) |
| 200(2차 폴백) | 약 31.5시간(1.3일) | 약 15.8시간(0.66일) |

## 판정 (G1 기준 ②: N건이 5~7일 안에 가능한가)

**가능 — 여유 있음 [확인, 8/20 Sobol 표본 기반 추정].** parallelism=2만으로도 N=410이 2.7일로 5~7일 예산의 절반 이하. parallelism=4까지 올리면 1.35일. 이번 8건은 전체 400+건 중 처음 8개(Sobol 저차원 표본이라 분포 편향은 크지 않음)라 표본 크기가 작다는 한계는 있으나, 추가 측정 없이 **N·K·depth_max·rays·grid_delta를 `configs/sim/base.yaml`로 동결하는 근거로 충분**하다고 판단.

## 다음 단계

1. `configs/sim/base.yaml` 작성: K=20, depth_max=50, rays=300, grid_delta=0.3, n_train/val/test/ood=300/40/40/20, n_golden=10, 파라미터 범위(`sampling.py` SPACE와 동일하게 기재)
2. parallelism은 2로 시작(검증된 조건), 필요 시 4로 상향 검토
3. 골든셋 선정 규칙(중심5·경계3·극단2) 알고리즘은 별도 결정 필요 — base.yaml 범위 밖, W3-06에서 구현
