# W3-12 (계획 외 추가, 사용자 제보로 발견): OOD ion_mean_energy 음수 버그 수정

- 날짜: 2026-11-07
- 배경: 사용자가 "OOD 범위에서 이온 에너지 값이 음수로 나오는 것 같다"는 제보.
  실측으로 확인한 결과 실제 버그였음(아래).
- 코드: `src/etchpipe/sim/sampling.py`(`make_ood_recipes`의 linear-scale 하한 확장 클램프)
- 테스트: `tests/unit/test_sampling.py::test_ood_linear_params_never_negative`(신규)
- 영향받아 재생성한 데이터: `data/raw/ood0004,0007,0008,0009,0010,0012,0013,0014,0015,0017,0018,0019`
  (12건), `data/recipes.json`의 해당 12건 `params.ion_mean_energy`

## 버그 원인 [확인]

`make_ood_recipes`가 linear-scale 파라미터를 학습 범위 아래로 확장할 때
`ext_lo = lo - (hi-lo)*(extend-1)`로 계산한다. `ion_mean_energy`(`lo=50, hi=200`,
`extend=1.5`, `configs/sim/base.yaml`)는 `span=(200-50)*0.5=75 > lo=50`이라
`ext_lo = -25`가 되어, **음수 구간을 포함해 균등 샘플링**된다. 이온 에너지(eV)는
물리적으로 음수가 될 수 없다.

실제 데이터(`data/splits.json`의 ood 20건) 확인: 5건(`ood0004,0008,0009,0015,0017`)이
음수(-6.4~-23.3)로 기록돼 있었다. `run_sim.py:49`가 `p.Ions.meanEnergy = params[...]`로
클램핑 없이 바로 ViennaPS에 주입하므로, 해당 5건은 시뮬레이션이 "실패"로 멈추진
않았지만(21/21 스냅샷) `final_depth_um`이 0.3~1.5로 거의 식각이 안 된 물리적으로
의미 없는 결과였다.

다른 linear 파라미터(`mask_width/height`, `lo=6, span=4, ext_lo=2>0`)는 문제없음.
log-scale 3개(`ion_flux/etchant_flux/oxygen_flux`)는 구조적으로 음수 불가.

## 수정 방법 [확인]

`ext_lo = max(lo - span, 1e-6)`로 하한을 0 바로 위로 클램프. linear-scale 파라미터
전체에 일반적으로 적용(향후 다른 파라미터가 같은 상황에 놓여도 안전).

### 재현성에 대한 추가 확인 [확인]

같은 seed(`20261027+1`)로 다시 뽑아도 "아래쪽" 분기를 타지 않은 파라미터·레시피는
완전히 동일한 값이 나온다(RNG 소비 횟수가 분기에 무관하게 동일하기 때문). 하지만
`ion_mean_energy`가 "아래쪽" 분기를 탄 ood 20건 중 12건은 구간이 `[-25,50)`에서
`[~0,50)`로 바뀌면서 **값 자체가 달라진다**(같은 난수 `u`를 다른 구간에 매핑하므로) —
음수였던 5건뿐 아니라 원래 양수였던 7건(`ood0007,10,12,13,14,18,19`)도 값이
바뀌었다. 사용자 확인 후(둘 다 보여주고 선택받음) "단일 seed 배치로 일관되게"
12건 전체를 재생성하는 쪽으로 진행했다(나머지 8건은 "위쪽" 분기라 완전히 동일,
재생성 불필요).

## 재생성 실행 [확인]

1. `data/recipes.json`의 12건 `params`를 수정된 `make_ood_recipes` 결과로 교체,
   kind PVC(`etch-data`)의 `/data/recipes.json`에 반영.
2. PVC의 기존 12건 raw 디렉터리 삭제(재시도 skip 로직 우회).
3. `etch-sim:0.1` 이미지로 임시 Pod(`sim-regen-w3-12b`) 실행, 1회성 스크립트
   `scripts/regen_ood_w3_12.py`로 12건만 `run_sim()` 재실행.
   (1차 시도는 Pod `sleep 3600` 타임아웃에 걸려 9/12에서 중단 — `sleep 7200`으로
   재생성, skip 로직 덕분에 나머지 3건만 이어서 완료.)
4. 결과를 `data/raw/`로 백업(`kubectl cp`), 로컬에서 `ion_mean_energy > 0` 및
   스냅샷 개수(`actual_k+1`) 확인.

```
ood0004  11.03   21 snapshots
ood0007  30.36   21
ood0008   1.14   21
ood0009   1.76   21
ood0010  47.64   21
ood0012  38.94   12  (stopped_early=True, final_depth_um=51.0, 정상 조기종료)
ood0013  23.95   21
ood0014  27.04   21
ood0015   4.43   21
ood0017  12.40   21
ood0018  49.67   13  (stopped_early=True, final_depth_um=52.5, 정상 조기종료)
ood0019  45.03   21
```

12건 전부 `ion_mean_energy > 0`, `viennaps=4.5.0`, `meta.json` 정상.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_sampling.py -q
8 passed   (기존 7건 그대로 통과 + 신규 회귀 테스트 1건)
```

## 영향 범위

- `data/raw/ood0004,07,08,09,10,12,13,14,15,17,18,19` 12건 재생성 완료, 로컬 백업 갱신.
- `data/recipes.json`도 동일 12건 수정.
- **GitHub Release(`data-v1-n400`)와 Hugging Face(`etch-ai-automation-data`)에는
  아직 수정 전(버그 포함) 버전이 올라가 있다** — 재업로드는 사용자 지시에 따라
  이후 단계(압축·재업로드)에서 처리할 예정.
- `data/processed/`는 아직 비어 있어(= `process_dataset`을 실제 데이터에 돌린 적
  없음) 영향 없음.
- +100건 생성 계획은 보류 결정(사용자 확인) — split 비율 변경이 필요해
  `project-definition.md` 수정 + DOC- 항목이 선행돼야 하고, 지금은 W3-04~W3-08을
  N=400(수정분 포함) 기준으로 먼저 완료하는 게 맞다고 판단.
