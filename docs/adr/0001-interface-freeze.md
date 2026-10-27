# ADR-0001: 2주차 말 인터페이스 동결

- 날짜: 2026-10-28
- 상태: 승인(2주차 완료 시점)
- CLAUDE.md 작업규칙 6: "인터페이스(스키마, 해시 규칙, 종료 코드)는 2주차 말에 동결. 변경은 ADR 작성 후에만."

이 문서 이후 아래 스키마를 바꾸려면 새 ADR을 작성하고 영향받는 체크리스트 항목에 근거를 남긴다.

## 1. `meta.json` (C1, `run_sim`)

```json
{
  "params": {"ion_flux": 0.0, "...": 0.0},
  "K": 20,
  "actual_k": 20,
  "step_min": 1.0,
  "rays": 300,
  "seed": 42,
  "grid_delta": 0.3,
  "depth_max": 50.0,
  "stopped_early": false,
  "final_depth_um": 0.0,
  "viennaps": "4.5.0",
  "wall_sec": 0.0,
  "params_hash": "0123456789ab"
}
```

- `params_hash` = `sha1(json.dumps(params, sort_keys=True))[:10]`
- `depth_max` 기본값 50µm(사용자 결정 2026-10-28, 근거: `reports/evidence/W1-S1_timing.md`)
- `stopped_early`/`actual_k`: 정지조건으로 K 전부를 못 돌았으면 `actual_k < K`

## 2. `dataset_hash` (C2, `etchpipe.etl.etl.compute_dataset_hash`)

```
dataset_hash = sha1("".join(sorted(run_id별 params_hash)))[:12]
```

- 레시피 집합이 하나라도 바뀌면(추가/제거/재생성) 값이 바뀐다.
- `manifest.json` 스키마: `{"dataset_hash": str, "theta_keys": [str,...], "n_recipes": int, "splits_sizes": {"train": int, "val": int, "test": int, "ood": int}}`
- `splits.json` 스키마: `{"train": [run_id,...], "val": [...], "test": [...], "ood": [...]}`

## 3. 게이트 종료 코드 (C5, `etchpipe.gate.gate.run_gate`)

| 코드 | 의미 |
|---|---|
| 0 | PASS |
| 1 | BLOCK — 성능 미달(골든 Chamfer 초과 또는 롤아웃 후반 오차 급증) |
| 2 | BLOCK — 데이터 무결성(`dataset_hash` 불일치 또는 golden hash 불일치, 둘 다 독립적으로 체크) |

`GateResult(passed: bool, exit_code: int, reasons: list[str])`.

## 4. 레지스트리 (C6, `etchpipe.registry.registry.Registry`)

- `history.jsonl`: 한 줄당 `{"model_id": str, "promoted": bool, "exit_code": int, "reasons": [str], "timestamp": ISO8601}`
- `current`: champion `model_id`를 담은 평문 파일(PASS일 때만 갱신)

## 5. 서빙 API (C7, `etchpipe.serve.app`)

- `GET /health` → `{"status": "ok"}` (API 키 불필요)
- `GET /model` → `{"model_version": str, "valid_range": ..., "gate_result": ...}` (API 키 필요)
- `POST /predict` 요청: `{"recipes": [{ion_flux, etchant_flux, oxygen_flux, ion_mean_energy, ion_exponent, mask_width, mask_height}, ...], "times": [float, ...]}`
- `POST /predict` 응답: `{"model_version": str, "results": [{"profiles": [{"t": float, "points": [[x,y],...]}], "cd": float, "depth": float, "taper_deg": float, "in_range": bool}, ...]}`
- API 키: `x-api-key` 헤더, 서버 기동 시 `ETCH_API_KEY` 환경변수 필수(없으면 기동 거부)
- 필수 키 누락/타입 오류 → 422(pydantic 자동 검증)

## 6. 아직 동결하지 않은 것

- SDF(부호 있는 거리장)의 정확한 좌표계·부호 규칙 — W3-03에서 확정 후 ADR-0002 예정
- `/predict` 응답의 `cd`/`taper_deg` 실제 계산 방법 — 현재는 더미(0.0 고정), W5-01에서 확정
- 게이트 규칙 ②③⑤(이전 모델 대비, 추론시간, 롤아웃 외) — W5-04/W5-05에서 확정
