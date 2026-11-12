# W4-02: 손실 Chamfer 구현·테스트, 학습 루프(전이쌍, 팀포싱)

- 날짜: 2026-11-09
- 코드: `src/etchpipe/train/chamfer_loss.py`(`chamfer_loss`, 신규),
  `src/etchpipe/train/loop.py`(`iter_transition_pairs`, `train_one_epoch`, 신규)
- 테스트: `tests/unit/test_chamfer_loss.py`(4건), `tests/unit/test_train_loop.py`(4건)

## 손실 [확인]

`chamfer_loss(pred, true)`: `eval/metrics.py`의 평가용 `chamfer_distance`(scipy cKDTree,
비미분)와 같은 정의(양방향 평균 최근접 거리)를 `torch.cdist`로 구현해 역전파가 되게 했다.
점 개수가 달라도 된다(`pred` N개, `true` M개) — 스냅샷마다 점 개수가 다른 문제(설계서 §4.1
위험)를 직접 변위 MSE가 아니라 Chamfer로 푸는 이유와 같다.

## 학습 루프 [확인]

- `iter_transition_pairs(graphs_root, run_ids)`: run마다 연속한 그래프 쌍 `(g_k, g_{k+1})`을
  순서대로 낸다. 입력은 항상 **디스크에 저장된 정답** 그래프에서 로드 — **팀포싱**(모델
  자신의 이전 예측을 다시 입력으로 쓰는 자기회귀가 아님. 그건 평가용 롤아웃, W4-05).
- `train_one_epoch(model, optimizer, graphs_root, run_ids)`: 전이쌍마다 `model(node,
  edge_index, edge)`로 변위(N,2)를 예측하고 `pos+변위`를 다음 스냅샷 점들과
  `chamfer_loss`로 비교, 개별 역전파(쌍마다 점 개수가 달라 배치로 묶기 어려움).
- 모델 인터페이스는 `model(node, edge_index, edge) -> displacement`로 고정했다(설계서 C3:
  `input_dim_nodes=14, input_dim_edges=3, output_dim=2`와 일치). 실제
  `nvidia-physicsnemo` MeshGraphNet의 `forward` 시그니처가 이와 다르면 W4-03에서 얇은
  래퍼로 맞춘다 [미확인: physicsnemo MeshGraphNet 정확한 입력 형식 — 호스트에
  `nvidia-physicsnemo`가 설치돼 있지 않아 이번엔 확인 못 함, train 컨테이너에서 W4-03
  착수 시 확인 필요].

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_chamfer_loss.py tests/unit/test_train_loop.py -v
8 passed
  test_matches_numpy_chamfer_distance_on_random_points  (eval/metrics.chamfer_distance와 오차<1e-5)
  test_zero_when_point_sets_identical
  test_gradient_flows_to_pred
  test_loss_decreases_as_pred_moves_toward_true
  test_iter_transition_pairs_counts_k_pairs_per_run   (실제 fixture 3 run x actual_k=3 = 9쌍)
  test_iter_transition_pairs_shapes
  test_train_one_epoch_reduces_loss_over_epochs       (토이 모델 20에폭, 손실 단조 감소 확인)
  test_train_one_epoch_updates_model_parameters

PYTHONPATH=src python -m pytest tests/unit -q
94 passed (기존 86 + 신규 8)

python -m ruff check src/etchpipe/train/ tests/unit/test_chamfer_loss.py tests/unit/test_train_loop.py
All checks passed!
```

학습 루프 테스트는 `tests/fixtures/raw`(실제 ViennaPS 결과, r0000~r0002, K=3)를
`etl_run`으로 그래프 변환해 사용 — 합성 데이터가 아니라 실제 시뮬레이션 산출물로 루프
플러밍을 확인했다. 단, 모델 자체는 **토이 모델**(`node` 특징만 보는 `nn.Linear`,
`edge_index`/`edge` 무시)이라 과적합·실제 MeshGraphNet 정확도 확인은 범위 밖(W4-03).

## 범위에서 제외한 것

- 실제 MeshGraphNet(`nvidia-physicsnemo`) 연결: 호스트 Python에 해당 패키지가 없어
  (train 컨테이너 전용) 이번 항목에서는 확인 불가 — W4-03에서 컨테이너 안에서 진행.
- 배치 처리: 쌍마다 점 개수가 달라 PyG 식 블록-대각 배치를 쓰려면 추가 설계가 필요 —
  지금은 쌍 1개씩 개별 역전파(스텝당 느릴 수 있음, 실측은 W4-04 본 학습에서).

## 추가: 입력 정규화 지원 (2026-11-09)

GPU 팀원이 전달한 MGN 초안 체크포인트(`runs/mgn_draft/ckpt_mgn.pt`)를 분석하는 과정에서
(아래 "W4-03 이관" 참고) 그 체크포인트가 `(node-mu_n)/sd_n`, `(edge-mu_e)/sd_e` 정규화된
입력으로 학습됐음을 확인 — 반면 당시 `train_one_epoch`은 정규화 없이 raw 특징을 그대로
모델에 넣고 있었다. 채널별 스케일 차이가 큼(예: theta 한 채널 평균 4944/표준편차 3230,
나머지는 O(1)) — 이것이 2026-11-09 CPU 과적합 프로브에서 손실이 0.75 선에 plateau된
원인일 가능성이 있다고 판단.

`src/etchpipe/train/loop.py`에 `NodeEdgeNorm`(dataclass), `load_norm_stats(stats_path)`,
`normalize_node`/`normalize_edge` 추가. `train_one_epoch(..., norm=None)` — 기본값
`None`이면 기존과 동일(하위 호환, 기존 테스트 변경 없이 통과). `norm`을 주면 모델 호출
전 node/edge만 정규화하고 출력 변위는 그대로 둠(체크포인트에 별도 출력 통계 없음).

정규화 소스는 `data/processed/<dataset_hash>/stats.json`(W3-04)을 쓴다 — 실제 대조
결과 팀원 체크포인트의 `mu_n/sd_n/mu_e/sd_e`와 거의 일치(독립적으로 같은 데이터에서
같은 값을 다시 계산한 것으로 보임), 교차 검증됨.

테스트: `tests/unit/test_train_loop.py`에 3건 추가(`test_load_norm_stats_reads_json`,
`test_normalize_node_and_edge_apply_mean_std`, `test_train_one_epoch_with_norm_runs_and_updates`).

```
PYTHONPATH=src python -m pytest tests/unit/test_train_loop.py -v  → 7 passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
  → 87 passed, 3 failed (test_run_train.py git_sha 관련 3건 — etch-train:0.1 이미지에
    git 미설치라 컨테이너 안에서만 실패, 오늘 변경과 무관한 기존 환경 한계 [미확인:
    Dockerfile.train에 git 추가 여부])
python -m ruff check src/etchpipe/train/ tests/unit/test_train_loop.py → All checks passed
```

## W4-03 이관 메모

실제 MeshGraphNet 연결까지는 이번 항목에서 끝냈고(`src/etchpipe/models/meshgraphnet.py`,
`docker/Dockerfile.train`), 소부분집합 과적합 테스트(원래 W4-03 범위)는 CPU 자원
부족(epoch당 150~290초)으로 GPU 가용 팀원에게 이관함 — 자세한 내용은
`docs/project-checklist.md` W4-03 항목 메모 참고.
