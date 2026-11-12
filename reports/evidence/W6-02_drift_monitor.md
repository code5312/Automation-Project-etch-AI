# W6-02: 드리프트 모니터 — 신규 레시피 10개 실측

- 날짜: 2026-11-12
- 코드: `src/etchpipe/retrain/monitor.py`(`mean_chamfer_over_recipes`/`check_drift`) +
  `scripts/run_drift_sim.py`(`etch-sim:0.1`)/`scripts/run_drift_monitor.py`(`etch-train:0.1`)
- champion: `runs/mgn_tuned/model.pt`(epoch=2, processor_size=6, hidden_dim_processor=128)
- 모니터링 레시피: `sampling.make_recipes(10, seed=9001, prefix="mon")` — 학습 Θ 범위 안
  (OOD 아님), live 설정(`substrate_a_ie=8.05`, `sim_config_id=live_v1`)으로 시뮬레이션

## 실행

```
# 1단계(etch-sim:0.1): PYTHONPATH=src python scripts/run_drift_sim.py
#   mon0000~mon0009 10개 전부 완료, 총 wall_sec ≈ 3350초(≈56분)
#   (mon0000: 557.3s/k20, mon0001: 440.0s/k10, ... mon0009: 349.9s/k20 — 조기종료 4건)
# 2단계(etch-train:0.1): PYTHONPATH=src python scripts/run_drift_monitor.py
```

## 결과 [확인]

```json
{
 "champion_golden_chamfer": 0.4227,
 "mean_chamfer": 0.4461,
 "threshold_mult": 1.5,
 "drift_threshold": 0.6341,
 "triggered": false
}
```

| 레시피 | Chamfer |
|---|---|
| mon0000 | 0.445 |
| mon0001 | 0.597 |
| mon0002 | 0.109 |
| mon0003 | 0.537 |
| mon0004 | 0.537 |
| mon0005 | 0.194 |
| mon0006 | 0.496 |
| mon0007 | 0.763 |
| mon0008 | 0.336 |
| mon0009 | 0.448 |

→ 전체 저장: `reports/eval/W6-02_drift_monitor.json`

## 해석 [제안]

- `Substrate.A_ie` ×1.15 드리프트에서는 **자동 반복 루프가 트리거되지 않았다**
  (0.4461 < 0.6341). champion이 이 정도 드리프트에는 재학습 없이도 충분히 버틴다는
  뜻으로 보인다 — mon0007(0.763)처럼 개별 레시피는 임계값을 넘지만, 평균(design.md가
  정의한 판정 기준)은 넘지 않는다.
- **W6-03/04(자동 반복 루프)에 대한 함의**: 이 드리프트 강도로는 루프를 "자연 발생"으로
  시연할 수 없다. 체크리스트 증거②가 요구하는 건 "루프 자체가 동작하는지"이므로,
  W6-04 구현 후에는 (a) 더 강한 드리프트로 다시 모니터링하거나 (b) 트리거를 수동으로
  걸어(`check_drift`를 거치지 않고 바로 루프 진입) 증거를 남기는 방법 중 하나가 필요하다
  — 사용자 결정 필요.
