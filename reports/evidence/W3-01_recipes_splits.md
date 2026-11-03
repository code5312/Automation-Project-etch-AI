# W3-01: make_recipes/make_splits 실행 → data/recipes.json, data/splits.json 생성

- 날짜: 2026-11-03
- 스크립트: `scripts/gen_splits.py` (seed=20261027, n_train/val/test/ood=300/40/40/20 — `configs/sim/base.yaml`과 동일)

## 사전 확인: 기존 probe8 실측과의 연속성 [확인]

`make_recipes(380, seed=20261027)`의 첫 8개(`r0000`~`r0007`)가 probe8(`W3-01_probe8_depthmax_kind.md`)에서
이미 실제 ViennaPS로 실행한 레시피와 파라미터가 완전히 일치하는지 확인:

```
PYTHONPATH=src python -c "... make_recipes(380, seed=20261027)[:8] vs PVC meta.json 실측값 비교 ..."
결과: 8건 전부 ion_flux 등 파라미터 완전 일치(ALL MATCH: True), rec20[:20]==rec380[:20] True
```

Sobol(scramble=True, seed 고정)은 요청 개수(n_total)가 바뀌어도 앞쪽 포인트를 그대로 보존한다는 것이
실측으로 재확인됨 — 이번에 400건으로 늘려도 이미 측정한 8건의 결과를 버리지 않아도 됨.

## 실행

```
PYTHONPATH=src python scripts/gen_splits.py
```

출력:
```
recipes: 400 -> data/recipes.json
splits sizes: {'train': 300, 'val': 40, 'test': 40, 'ood': 20} -> data/splits.json
```

## 검증

- `data/splits.json` 스키마가 ADR-0001과 일치(`{"train":[...], "val":[...], "test":[...], "ood":[...]}`, run_id만 포함).
- train/val/test 교집합 0, ood와 train 교집합 0 [확인, 직접 조회].
- `data/recipes.json` 400건, run_id 중복 0건 [확인].
- `PYTHONPATH=src python -m pytest tests/unit/test_sampling.py -q` → 7 passed [확인].

## 남은 것

- **golden(10)**: 선정 규칙(중심5·경계3·극단2) 알고리즘 미정 — `splits.json`에 키를 추가하지 않음(ADR-0001 동결 스키마 위반 방지). W3-06에서 ADR 또는 별도 파일로 처리.
- **실제 시뮬레이션(W3-02)**: 지금 만든 건 레시피 파라미터 할당(Sobol 샘플링)일 뿐, ViennaPS 실행은 아직 안 함. `data/raw/`에는 probe8(r0000~r0007, 실측 완료) 외 나머지 392건의 raw 데이터가 없음.
