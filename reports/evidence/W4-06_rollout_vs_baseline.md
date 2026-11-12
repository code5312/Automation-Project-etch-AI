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

## 업데이트(2026-11-11): `mgn_tuned`(ep2, 팀원 최종 후보)로 재실행 [확인 — ADR-0004 새 정의]

ADR-0004로 Chamfer 정의가 바뀌었고(`eval/metrics.py::chamfer_distance`, 점-선분+
densify+평균÷2), 거기에 더해 `train/rollout.py::rollout_val_error_by_step`에 outer_path
추출 누락 버그를 고치고(위 W3-07 업데이트와 같은 원인), `mgn_tuned/inference.py`의
마스크 고정·교차가드·고리제거를 `rollout()`에 포팅(`fix_mask`/`guard`/`loop_removal`
신설, 기본 꺼짐)했다. 위 표의 "초안(epoch=5)"/"본 학습(epoch=99)" 수치는 전부 **구
Chamfer 정의 + outer_path 미적용 + 마스크고정/가드 없음** 기준이라 직접 비교는 안 되고,
`mgn_tuned`만 새 기준으로 따로 측정했다:

```
PYTHONPATH=src python -m etchpipe.eval.eval runs/mgn_tuned/model.pt \
  --config configs/model/mgn_v1.yaml --out-dir reports/eval/mgn_tuned_val40_guarded \
  --device cpu --steps 5,10,20 --n-plots 2 --fix-mask --guard --loop-removal
```

| | Chamfer 평균 | Chamfer 중앙값 | 깊이오차 평균(µm) | 깊이오차 중앙값(µm) | 선 꼬임(합) |
|---|---|---|---|---|---|
| (a) 변위 0(새 정의+outer_path, W3-07 2차) | 8.143 | 9.620 | 40.964 | 49.793 | - |
| (b) 깊이 MLP 외삽(새 정의+outer_path, W3-07 2차) | 0.438 | **0.351** | 2.397 | 1.910 | - |
| 모델 A 롤아웃(**mgn_tuned, ep2**) | **0.373** | 0.388 | **1.126** | **0.905** | **0** |

- 깊이오차: 베이스라인(b)보다 뚜렷하게 낫다(평균 1.126 vs 2.397, 중앙값 0.905 vs 1.910).
- 선 꼬임 0건 — 팀원 골든셋 보고(0건)와 일치, 교차가드+고리제거 포팅이 제대로 동작함을
  보여준다.
- **Chamfer는 평균 기준으로는 베이스라인(b)보다 낫고(0.373<0.438), 중앙값 기준으로는
  근소하게 더 나쁘다(0.388>0.351, 차이 0.037µm)** — W3-07이 "평균/중앙값 중 뭘 D4
  기준으로 할지는 W4에서 결정"이라고 남겨둔 질문. **사용자 결정(2026-11-11): 평균을
  기준으로 한다.** → **D4 판정: 통과**(0.373 < 0.438). W4-07(롤아웃 v2)은 "D4 실패 시"
  조건이라 착수 불필요.
- **알려진 한계(측정 완료, 2026-11-11) — 생략 영향이 작지 않음**: `mgn_tuned/inference.py`의
  "점 채우기"(선분이 0.42µm보다 길어지면 중간에 점을 추가하는 재메싱)는 포팅하지
  않았다(우리 `rollout()`의 "연결 유지"(점 개수·순서 불변) 설계와 충돌). 실제로
  val 40개 전부를 롤아웃해 선분 길이를 측정한 결과 [확인]:
  - t=0 선분 최대 길이: 0.3~0.42µm(정상, 학습 밀도와 일치).
  - **최종 스텝 선분 최대 길이: 3.7~25.4µm**(임계값의 최대 ~60배).
  - **40개 레시피 전부**에서 임계값(0.42µm) 초과 선분 발생(레시피당 17~33개, 전체
    선분의 약 8~15%).
  - 즉 "가끔 필요한 보정"이 아니라 롤아웃이 몇 스텝만 지나도 거의 항상 발생하는
    상황이다 — 그 구간에서는 모델이 학습 때 못 본 엣지 길이를 입력으로 받고, 실제
    표면 곡률(트렌치 바닥 모서리 등)이 직선으로 뭉개진다. 위 Chamfer 평균(0.373µm)이
    양호하게 나온 건 전체 선분의 85~92%가 여전히 정상 범위라 평균이 크게 안 흔들린
    것으로 보이며, 국소적으로는 상당히 깨져 있을 수 있다. 스텝 수가 길어질수록(누적)
    악화될 가능성이 높음[제안: 해석, 직접 검증 안 함].
  - 포팅하려면 `rollout()`의 "연결 유지" 가정 자체를 바꿔야 해서(점 개수가 스텝마다
    변함) 작업 범위가 크다 — 영향 범위를 A~F로 정리해 검토함(반환 타입 변경, 신규 점
    법선·재료 보간, 메시 증식에 따른 비용 증가, 스텝 내 순서, `edge_index` 구성 규칙
    유지, 호출부 전체 수정).

  **최종 결정(사용자, 2026-11-11): 포팅하지 않는다.** 근거: ① D4는 이미 포팅 없이
  통과했고 선 꼬임도 0건 — 어떤 게이트·체크리스트 판정도 이걸로 막혀 있지 않다(순수
  정확도 개선용). ② 모델은 이미 동결(더 개선 안 함) 결정과 방향이 안 맞는다. ③
  CLAUDE.md/설계서가 명시한 프로젝트 우선순위("게이트·재학습 자동화가 핵심, 정확도
  극대화는 후순위")와 안 맞고, 포팅 범위(A~F)가 남은 주차(W5~W8) 작업량에 비해 과함.
  ④ eval(`eval.py`)과 서빙(W6-08 예정)이 같은 롤아웃 코드를 쓰게 유지하는 게 "평가 때만
  특별 처리"하는 평가-서빙 불일치보다 구조적으로 더 건전하다. **알려진 잔존 한계**로
  남긴다: 선분이 심하게 늘어나는(최대 25µm) 레시피는 국소적으로 단면이 뭉개질 수 있어,
  대시보드·데모에서 특정 레시피를 보여줄 때 시각적으로 어색할 수 있음(집계 지표에는
  안 잡힘) — W7(대시보드)/W8(데모) 단계에서 레시피 선택 시 참고.

## 업데이트(2026-11-11): 체크리스트 W4-05(GPU 팀원 val 40개 오차 기록) 충족 근거로 사용

위 `mgn_tuned` 행이 W4-05가 요구하는 "실제 학습된 체크포인트 기준 val 40개 오차 기록"에
해당한다 — `rollout_val_error_by_step`로 직접 산출.
