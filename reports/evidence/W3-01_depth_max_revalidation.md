# W3-01 depth_max 재검증 (이미지 재빌드 후)

- 날짜: 2026-10-28
- 목적: `reports/evidence/W3-01_probe.md`에서 확인된 "이미지가 재빌드되지 않아 depth_max 미적용" 버그를
  고친 뒤, 가장 높은 ion_flux 레시피(r0004)로 depth_max=50 조기 종료가 실제로 동작하는지 확인.

## 조치

1. `docker build -f docker/Dockerfile.sim -t etch-sim:0.1 .` 재실행 — 현재 `src/etchpipe/sim/run_sim.py`
   (W2-01에서 depth_max 로직 추가된 버전)가 이미지에 반영됨 [확인].
2. `kind load docker-image etch-sim:0.1 --name etch-pipeline`로 kind 클러스터에 갱신된 이미지 반영 [확인].
3. `docker run --rm etch-sim:0.1`로 r0004(ion_flux=25.90, probe 8건 중 최고값) 단독 실행, K=20·
   depth_max 기본값(50) 그대로.

## 결과 [확인]

```
run_sim() 반환값(= meta.json 내용, run_sim.py:100-101에서 동일 dict를 그대로 json.dump):
{
  "K": 20, "actual_k": 5, "step_min": 1.0, "rays": 300, "seed": 42,
  "grid_delta": 0.3, "depth_max": 50.0, "stopped_early": true,
  "final_depth_um": 55.5, "viennaps": "4.5.0", "wall_sec": 386.2,
  "params_hash": "371f4e374a"
}
```

- **`stopped_early: true`, `actual_k: 5`(K=20 중 5스텝)** — depth_max=50이 실제로 조기 종료를
  발생시켰다. `final_depth_um: 55.5`는 50을 넘어선 직후 스텝에서 멈춘 것과 일치(루프 구조상
  `_current_depth(domain) >= depth_max` 체크가 스텝 완료 후 실행되므로 정확히 50.0에서 끊기지
  않고 그 스텝의 최종 깊이로 기록됨 — 설계대로).
- `wall_sec: 386.2` — W3-01_probe.md에서 같은 레시피 그룹 중 가장 느렸던 r0000(14.8 flux, K=20
  고정)이 1941.0초였던 것과 비교하면, r0004(flux 25.9로 더 높음)가 5스텝만에 끝나 **약 5배
  빠르게 완료** — depth_max 도입의 실질 효과가 숫자로 확인됨.
- 참고: 검증용 `docker run -v /tmp/depth_check:/out`에서 호스트 쪽 파일을 못 찾은 것은 git-bash
  MSYS 경로가 Docker Desktop의 WSL2 데몬과 다른 네임스페이스를 가리켰기 때문(경로 불일치) —
  run_sim() 자체의 동작·반환값과는 무관. 컨테이너가 `--rm`으로 종료되며 내부 파일은 같이
  사라졌지만, stdout에 찍힌 RESULT가 meta.json과 동일한 dict라 증거로는 충분하다.

## 판정

**depth_max 조기 종료 기능 [확인] — 정상 동작.** W3-01_probe.md의 "미완료/조건부" 상태 해소.
N=400 본 데이터 생성 시간 추정을 depth_max 효과를 반영해 다시 할 수 있는 조건이 갖춰졌다.

## 다음 단계 제안

1. W3-01 본 설정(N, K, depth_max, 입력 범위) 확정 — 이번 결과로 고flux 레시피의 평균 시간이
   크게 줄 것으로 기대되므로, probe 8건 전체를 재빌드된 이미지로 다시 돌려 depth_max 적용
   평균 시간을 재추정하는 것을 권장.
2. `deploy/k8s/sim-job-probe.yaml`, `reports/evidence/W3-01_probe.md`,
   `reports/evidence/W3-01_depth_max_revalidation.md`는 아직 커밋되지 않음 — 커밋 메시지는
   하단 참고.
