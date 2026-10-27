# W1-S5: MaterialIds 0/10 재료 매핑 확정

- 목표: `.vtp` 스냅샷의 `MaterialIds` 셀 데이터에서 0과 10이 각각 어떤 재료(Si/마스크)인지 확정 → `configs/etl/materials.yaml`
- 시간 상자: 1시간
- 선행 조건: S1(레시피 20개 실행) 전체가 필요한 건 아니고, 스냅샷 1건만 있으면 되므로 레시피 1건(K=2)을 먼저 실행했다(시드 20261027).

## 실행 명령

```
# 1) 레시피 1건 생성·실행 (reports/evidence/W1-S5_probe.py)
docker run --rm -v "$PWD:/work" -w /work python:3.13-slim bash -c "
  apt-get install -y libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6
  pip install -r requirements-sim.txt pyvista
  python reports/evidence/W1-S5_probe.py"

# 2) MaterialIds 대조 (reports/evidence/W1-S5_inspect.py)
docker run --rm -v "$PWD:/work" -w /work python:3.13-slim bash -c "
  apt-get install -y libgl1 libxrender1 libxext6 libsm6 libice6 libx11-6
  pip install pyvista numpy
  python reports/evidence/W1-S5_inspect.py"
```

(두 스크립트 모두 pyvista를 임시로 설치해 쓴다 — `requirements-sim.txt`에는 넣지 않는다. 이유는 W1-03 증거: ETL/pyvista는 train 환경 담당이고, 이번 건 S5 조사용 임시 조회일 뿐이다.)

## 측정값

레시피 `r0000`(시드 20261027, 1건), `mask_height≈13.873`, K=2(약 72.5초, 100 rays):

| 스냅샷 | MaterialIds | 셀 수 | y 범위 | 평균 y |
|---|---|---|---|---|
| t000(식각 전) | 0 | 212 | -0.000 ~ 13.873 | 6.367 |
| t000(식각 전) | 10 | 42 | -0.000 ~ 0.150 | 0.007 |
| t002(2분 식각 후) | 0 | 210 | -0.000 ~ 13.872 | 6.428 |
| t002(2분 식각 후) | 10 | 92 | -7.216 ~ -0.150 | -5.154 |

- `MaterialIds=0`의 y 범위가 식각 전후로 거의 변하지 않고(6.367→6.428) 상한이 레시피의 `mask_height`(13.873)와 거의 일치 → **마스크**(식각되지 않는 재료).
- `MaterialIds=10`은 식각 전엔 y≈0의 얇은 표면선(Si 윗면, etl.py 주석대로 "Si 윗면이 y=0")이었다가, 식각 후 y가 -7.2까지 내려감(식각 깊이만큼 깎임) → **Si**(식각되는 재료).

## 판정

**확정**: `MaterialIds=0 → mask`, `MaterialIds=10 → Si`. `configs/etl/materials.yaml`에 기록.

**중요**: seed의 `etl.py` 주석은 `# 2: 재료 2종 [가정] 0=Si, 10=마스크`로 **반대로** 적혀 있었다(가정이었고 틀렸음이 이번에 확인됨). `etl.py` 내용은 이번 작업에서 고치지 않았다(지시: 이동만, 수정 금지) — **W3-03(ETL 완성)에서 반드시 고쳐야 할 항목**으로 남긴다. 지금 상태로 그대로 쓰면 그래프 노드 특성의 재료 원-핫 인코딩이 뒤집힌다.

## 대안

불필요(확정됨).
