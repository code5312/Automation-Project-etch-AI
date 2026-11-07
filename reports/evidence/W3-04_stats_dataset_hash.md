# W3-04: 정규화 통계 stats.json, dataset_hash, 해시 일치 테스트

- 날짜: 2026-11-07
- 코드: `src/etchpipe/etl/etl.py`(`compute_stats` 신규, `process_dataset`이 `splits["train"]`
  만으로 호출해 `stats.json` 작성)
- 테스트: `tests/unit/test_etl.py`
  - `test_compute_stats_matches_analytic_mean_std`(해석적으로 답을 아는 입력, 설계서 §9)
  - `test_compute_stats_accumulates_across_runs`
  - `test_process_dataset_stats_uses_train_split_only`(같은 run_id 집합이라도 train에
    더 포함시키면 stats가 달라짐을 확인 → **train에서만 계산됨**을 실증)
  - `test_process_dataset_stats_json_is_deterministic`(FR-05: 같은 입력(raw+splits)
    → 같은 `dataset_hash`·같은 `stats.json` 바이트)
  - 기존 `test_process_dataset_writes_manifest_and_splits`에 `stats.json` 존재 확인 추가

## 설계 [확인/제안]

`compute_stats(graphs_root, run_ids)`는 그래프 npz(`node`: 14채널, `edge`: 3채널)와
SDF npy를 1-pass(E[x], E[x^2] 누적)로 돌며 채널별 평균/표준편차를 계산한다
(`run_id` 전체를 한 번에 메모리에 올리지 않기 위함 [제안: N=400 규모에서 1-pass 자체
근거는 측정 안 함, 설계상 선택]). SDF는 채널이 아니라 격자 전체 값 하나로 취급해
전역 스칼라 평균/표준편차 하나만 낸다 [제안: 공간적으로 변하는 정규화(픽셀별)는
설계서에 요구사항이 없어 보류].

`process_dataset`은 전체 run_id를 ETL 변환한 뒤 `splits["train"]`만 넘겨
`compute_stats`를 호출한다 — FR-05 "정규화 통계는 train에서만 계산" 요구사항 그대로.

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_etl.py -q
20 passed   (기존 16건 전부 통과 + 신규 4건)

PYTHONPATH=src python -m pytest tests/unit -q
64 passed   (전체 스위트 회귀 없음)

python -m ruff check src/etchpipe/etl/etl.py tests/unit/test_etl.py
All checks passed!
```

## 실제 N=400 데이터 실행 [확인]

W3-11(SDF 내부 경계)·W3-12(OOD 음수 에너지) 두 버그 수정이 반영된 뒤, `data/processed/`가
비어 있던 상태에서 **처음으로** 실제 데이터에 `process_dataset`을 실행:

```
PYTHONPATH=src python -c "
import json
from etchpipe.etl.etl import process_dataset, THETA_KEYS
splits = json.load(open('data/splits.json'))
manifest = process_dataset('data/raw', splits, 'data/processed', THETA_KEYS, max_workers=4)
print(json.dumps(manifest, indent=1))
"
```

결과(소요 8분 20초):

```json
{
 "dataset_hash": "4b8c0ce1bc98",
 "theta_keys": ["ion_flux","etchant_flux","oxygen_flux","ion_mean_energy","ion_exponent","mask_width","mask_height"],
 "n_recipes": 400,
 "splits_sizes": {"train": 300, "val": 40, "test": 40, "ood": 20}
}
```

`data/processed/4b8c0ce1bc98/stats.json`(train 300건 기준, node 1,894,899개·edge
3,779,636개·sdf 168,460,288개 값):

| 채널 | 의미 | mean | std |
|---|---|---|---|
| node[0] | x | 0.0010 | 7.5763 |
| node[1] | y | -4.6345 | 13.7307 |
| node[2] | normal x | 0.0000 | 0.7531 |
| node[3] | normal y | 0.1158 | 0.6477 |
| node[4] | mask 원핫 | 0.5452 | 0.4980 |
| node[5] | Si 원핫 | 0.4548 | 0.4980 |
| node[6] | theta: ion_flux | 10.9067 | 7.0545 |
| node[7] | theta: etchant_flux | 4983.8014 | 3250.4773 |
| node[8] | theta: oxygen_flux | 944.3453 | 600.0680 |
| node[9] | theta: ion_mean_energy | 123.2734 | 43.0963 |
| node[10] | theta: ion_exponent | 389.3378 | 247.5558 |
| node[11] | theta: mask_width | 9.9124 | 2.3185 |
| node[12] | theta: mask_height | 10.1339 | 2.3074 |
| node[13] | t | 9.6915 | 5.7009 |
| edge[0] | rel x | 0.0000 | 0.1923 |
| edge[1] | rel y | 0.0000 | 0.2143 |
| edge[2] | norm | 0.2809 | 0.0629 |
| sdf | 전역 | -9.6697 | 14.1816 |

mask/Si 원핫 평균이 54.5:45.5로 합리적인 비율이고, `theta_*` 평균·표준편차가
`configs/sim/base.yaml`의 샘플링 범위(`ion_mean_energy` 50~200 등)와 정합적임을
눈으로 확인 [확인].

## 영향 범위

- `data/processed/4b8c0ce1bc98/`에 400건 전체 `graphs/`, `manifest.json`,
  `splits.json`, `stats.json` 생성됨(로컬, `data/`라 커밋 대상 아님).
- 이 결과물은 아직 GitHub Release·Hugging Face에 업로드하지 않음(raw 백업만
  되어 있는 상태, processed 산출물 업로드는 사용자 지시 시 진행).
