# W4-06: 롤아웃 오차 vs 베이스라인(a)(b) 비교표

- 날짜: 2026-11-09
- 코드: `scripts/run_rollout_comparison.py`(신규)
- 데이터: `reports/evidence/W4-06_rollout_vs_baseline.json`

## [CPU/매핑] 범위 — 비교 스크립트 준비 + 실제 실행

W3-07 베이스라인과 같은 틀(val 40건, t=0 -> t=actual_k 최종 표면, `eval/metrics.chamfer_distance`/
`depth_error`)로 모델 A **롤아웃**(W4-05) 오차를 비교하는 스크립트를 작성했다. 체크포인트
경로를 인자로 받아, 없으면 모델 A 행을 비워 두고 베이스라인만 기록하도록 짰다 — 원래는
"스크립트만 준비, 실제 수치는 W4-04 완료 후"로 계획했는데, GPU 팀원이 이미 넘겨준 MGN 초안
체크포인트(`runs/mgn_draft/ckpt_mgn.pt`, epoch=5, 미수렴)가 있어서 그걸로 실제로 한 번
끝까지 돌려봤다.

## 이 과정에서 발견·수정한 버그 2건 [확인]

체크포인트를 실제로 로드해보기 전까지는 드러나지 않았던 버그라, 비교표를 "준비만" 하고
멈췄다면 못 찾았을 것들이다.

1. **`processor_size` 재오판 — 정정**: W4-03에서 체크포인트의 `processor.processor_layers`가
   12개임을 보고 `processor_size=12`로 바꿨었다(`src/etchpipe/models/meshgraphnet.py`,
   `configs/model/mgn_v1.yaml`, `docs/project-checklist.md`에 반영). 이번에 실제
   `load_state_dict`를 돌려보니 `Missing key(s): processor.processor_layers.12~23...`로
   실패 — physicsnemo 소스(`meshgraphnet.py`)를 다시 보니
   `layers = list(chain(*zip(edge_blocks, node_blocks)))`로 edge_block·node_block을
   번갈아 쌓아서 `processor_layers` 개수가 **`2*processor_size`**다. 즉 레이어 12개는
   `processor_size=6`이 맞다(원래 제안값과 같았음) — `processor_size=12`로 바꾼 건
   틀린 판단이었다. 관련 파일(`meshgraphnet.py`, `mgn_v1.yaml`, `project-checklist.md`,
   `W4-01_train_script_skeleton.md`)을 모두 6으로 되돌리고 정정 기록을 남겼다.
2. **인코더/디코더 `hidden_dim_*` 미전달**: `processor_size=6`으로 고쳐도 여전히
   `size mismatch`(64 vs 128)가 나서 재확인 — `MeshGraphNetModel`이 physicsnemo
   `MeshGraphNet`에 `hidden_dim_processor`만 넘기고 있었는데, 실제로는
   `hidden_dim_node_encoder`/`hidden_dim_edge_encoder`/`hidden_dim_node_decoder`가
   **별도 인자**(기본값 각각 128)라 전달하지 않으면 인코더/디코더가 128로 만들어진다
   (체크포인트는 전부 64). `meshgraphnet.py`에서 이 셋도 `hidden_dim_processor`와 같은
   값으로 명시해 전달하도록 수정 — 이후 `load_state_dict` 성공.

## 실제 실행 결과 [확인, 단 아래 "주의" 참고]

```
PYTHONPATH=src python scripts/run_rollout_comparison.py
```

| | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) |
|---|---|---|---|---|
| (a) 변위 0 | 14.464 | 16.854 | 40.964 | 49.793 |
| (b) 깊이 MLP 외삽 | 8.552 | 9.133 | 2.397 | 1.910 |
| 모델 A 롤아웃(초안, epoch=5) | **1.282** | **1.335** | 2.173 | **1.119** |

## 주의 [미확인]

- 이 체크포인트는 **epoch=5, 미수렴 초안**이다(W4-03 이관 메모 참고). 그럼에도 베이스라인
  (a)(b)보다 Chamfer·깊이오차 모두 나은 것으로 나왔는데, 이게 실제 모델 성능인지 아니면
  val 분할이 train과 겹치거나(이 체크포인트가 어떤 train 분할로 학습됐는지 우리 쪽에서
  재확인 못함) 다른 이유로 낙관적인지는 **검증하지 않았다** — 게이트·최종 판정 근거로 쓰면
  안 된다. W4-04 본 학습(GPU 팀원, 정식 `processor_size=6`) 체크포인트가 나오면 같은
  스크립트로 재실행해 다시 확인한다.
- D4 판정("val 최종 스텝 Chamfer가 베이스라인(b)보다 작으면 통과") 자체를 이 수치로
  내리지 않는다 — 이 항목(W4-06)은 "비교표 준비"이고, D4는 체크리스트에 따로 명시된
  주 말 판정이다.

## 업데이트 (2026-11-10): W4-04 본 학습(epoch=99, GPU) 체크포인트로 재실행 [확인]

사용자가 Colab GPU로 `train_main.py`(W4-04)를 돌려 100에폭 전부 완료한 체크포인트
(`runs/20261110T021927908408_b369f6/ckpt_last.pt`, train loss 0.188→0.031)를 받았다.
재실행 전 실제 키 구조를 처음 확인하다 버그 1건 더 발견: `train_main.py`가
`MeshGraphNetModel` 래퍼 전체를 `model.state_dict()`로 저장해서 키에 `net.` 접두사가
붙는데(팀원 초안은 접두사 없는 physicsnemo 원본 형식), `_load_model_a`가 접두사 없는
키만 가정하고 있었다 — `net.` 접두사를 벗겨내도록 수정(두 형식 다 지원).

```
PYTHONPATH=src python scripts/run_rollout_comparison.py \
  "runs/20261110T021927908408_b369f6/ckpt_last.pt"
```

| | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) |
|---|---|---|---|---|
| (a) 변위 0 | 14.464 | 16.854 | 40.964 | 49.793 |
| (b) 깊이 MLP 외삽 | 8.552 | 9.133 | 2.397 | 1.910 |
| 모델 A 롤아웃(초안, epoch=5, 위 결과) | 1.282 | 1.335 | 2.173 | 1.119 |
| 모델 A 롤아웃(**본 학습, epoch=99**) | 2.182 | **2.383** | 3.007 | 2.521 |

**D4 판정("val 최종 스텝 Chamfer < 베이스라인(b)")은 Chamfer 기준으로는 통과**(중앙값
2.383 < 9.133, 평균도 마찬가지로 베이스라인(a)(b) 둘 다보다 낮음).

## 주의 2: 본 학습이 초안보다 두 지표 다 더 나쁘게 나옴 [미확인, 추가 조사 필요]

- 깊이오차는 베이스라인(b)보다 **더 나쁘다**(중앙값 2.521 > 1.910) — D4 판정 기준(Chamfer)
  자체는 통과지만, 모든 지표에서 베이스라인을 이기는 건 아니다.
- 더 눈에 띄는 건, epoch=5 미수렴 초안(팀원 제공)이 epoch=99까지 다 돌린 본 학습보다
  Chamfer·깊이오차 **둘 다** 더 좋게 나왔다는 점이다(1.335 vs 2.383, 1.119 vs 2.521).
  train loss는 본 학습이 훨씬 낮았는데(0.031 vs 초안 쪽은 알 수 없음) 롤아웃 val 성능은
  거꾸로다. 원인 후보(검증 안 됨):
  - 본 학습 쪽에 **val 기준 체크포인트 선택이 없다**(`train_main.py`는 매 에폭 `ckpt_last.pt`를
    덮어쓰기만 함) — epoch=99가 과적합된 지점일 수 있고, 더 이른 에폭이 val 기준으로는
    나았을 수 있다. 지난번 사용자가 지적한 "val 안 넣었는데 loss 떨어지는 게 의미 있나"가
    여기서 실제로 드러난 것으로 보인다.
  - 초안 체크포인트가 어떤 train 분할·하이퍼파라미터로 학습됐는지 우리 쪽에서 재확인 못함
    (기존 "주의" 항목 그대로 유효) — 서로 다른 학습 조건이라 직접 비교가 공정하지 않을 수 있음.
  - `lr=0.0001`(고정, 재검토 전 기본값) 등 하이퍼파라미터 차이.
- **다음 조치 제안**: `train_main.py`에 val 평가(teacher-forcing 또는 롤아웃) + `ckpt_best.pt`
  저장을 추가해, 에폭별로 val 성능이 실제로 어떻게 움직였는지 재학습하며 확인. 지금 당장
  D4를 "최종 판정"으로 확정하지 않는다 — 사용자 확인 필요.

## 테스트

```
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
  → 93 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관)
python -m ruff check src/ tests/ scripts/run_rollout_comparison.py → All checks passed
```
