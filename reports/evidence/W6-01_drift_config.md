# W6-01: `configs/sim/live.yaml` 드리프트 설정 + `sim_config_id`

- 날짜: 2026-11-12
- 코드: `src/etchpipe/sim/run_sim.py`에 `substrate_a_ie`/`sim_config_id` 인자 추가
  (기본 `substrate_a_ie=None` → ViennaPS 기본값 7.0 그대로, 하위 호환), `configs/sim/live.yaml` 신설

## 사전 확인 [확인]

```
etch-sim:0.1 컨테이너에서:
>>> p = ps.SF6O2Etching.defaultParameters()
>>> p.Substrate 속성: A_ie, A_sp, B_ie, B_sp, Eth_ie, Eth_sp, beta_sigma, k_sigma, rho
>>> p.Substrate.A_ie == 7.0  (기본값, 정의서 §3 증거②와 일치)
>>> viennaps.__version__ == '4.5.0'
```

## 스모크 테스트 [확인]

```
run_sim(params, out, K=3, substrate_a_ie=8.05, sim_config_id="live_v1")
-> meta.json: {"substrate_a_ie": 8.05, "sim_config_id": "live_v1", "final_depth_um": 9.0,
               "wall_sec": 102.5, "stopped_early": false, ...}
```
오류 없이 끝까지 돌았고, 오버라이드 값이 `meta.json`에 그대로 기록됨을 확인.

## 테스트

```
python -m ruff check src/etchpipe/sim/run_sim.py configs/sim/live.yaml 경로는 ruff 대상 아님(yaml)
→ run_sim.py: All checks passed
```
run_sim.py는 ViennaPS가 필요해 기존 관례대로 단위 테스트 없음(실행 평가로 검증,
`sim_batch.py`/`sampling.py`와 같은 패턴) — 위 스모크 테스트가 그 역할.
