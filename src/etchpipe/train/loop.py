"""C3: 학습 루프(W4-02) — 전이쌍(t_k -> t_{k+1}) + 팀포싱.

"팀포싱": 각 스텝의 입력은 항상 **정답** 그래프(t_k)에서 가져온다(모델 자신의 이전 예측을
다시 입력으로 쓰는 자기회귀 롤아웃이 아니다) — 평가 단계의 다단계 롤아웃(W4-05)과는 다른
학습 방식. 모델은 입력 그래프의 각 점에 대한 변위(dx,dy)를 예측하고, `pos+변위`를
`chamfer_loss`로 다음 스냅샷(t_{k+1})의 점들과 비교한다(점 개수가 달라도 됨).

모델 인터페이스: `model(node, edge_index, edge) -> displacement`
  node: (N,14) float32, edge_index: (2,2E) int64, edge: (2E,3) float32, 반환: (N,2) float32.
실제 MeshGraphNet(nvidia-physicsnemo) 연결 시 이 시그니처가 맞지 않으면 W4-03에서 얇은
래퍼로 맞춘다[미확인: physicsnemo MeshGraphNet.forward의 정확한 입력 형식].

입력 정규화(`NodeEdgeNorm`): MeshGraphNet에 넣기 전 node/edge 특징을
`(x-mean)/std`로 정규화하는 게 표준(채널별 스케일 차이가 커서 — 예: theta 채널 하나는
평균 수천 단위, 나머지는 O(1) — 정규화 없이 학습하면 손실이 잘 안 떨어짐, 2026-11-09
CPU 과적합 프로브에서 실측). `data/processed/<dataset_hash>/stats.json`(W3-04, train
기준)을 그대로 쓴다 — 외부에서 받은 MGN 초안 체크포인트(`runs/mgn_draft/ckpt_mgn.pt`)가
독자적으로 계산한 `mu_n/sd_n/mu_e/sd_e`도 이 stats.json과 거의 일치함을 확인함(같은 데이터,
같은 채널 순서라는 교차 검증). `norm=None`(기본값)이면 정규화 없이 기존과 동일하게 동작
(하위 호환).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from etchpipe.train.chamfer_loss import chamfer_loss


class GraphDisplacementModel(Protocol):
    def __call__(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass
class NodeEdgeNorm:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor


def load_norm_stats(stats_path) -> NodeEdgeNorm:
    """`stats.json`(W3-04, `process_dataset` 출력)에서 node/edge 정규화 통계를 읽는다."""
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    return NodeEdgeNorm(
        node_mean=torch.tensor(stats["node_mean"], dtype=torch.float32),
        node_std=torch.tensor(stats["node_std"], dtype=torch.float32),
        edge_mean=torch.tensor(stats["edge_mean"], dtype=torch.float32),
        edge_std=torch.tensor(stats["edge_std"], dtype=torch.float32),
    )


def normalize_node(node: torch.Tensor, norm: NodeEdgeNorm) -> torch.Tensor:
    return (node - norm.node_mean) / norm.node_std


def normalize_edge(edge: torch.Tensor, norm: NodeEdgeNorm) -> torch.Tensor:
    return (edge - norm.edge_mean) / norm.edge_std


def load_graph(path) -> dict[str, torch.Tensor]:
    """그래프 npz 1개(`etl.to_graph` 출력) 로드. 롤아웃(`train/rollout.py`)에서도 재사용."""
    z = np.load(path)
    return {
        "node": torch.from_numpy(z["node"]),
        "edge_index": torch.from_numpy(z["edge_index"]),
        "edge": torch.from_numpy(z["edge"]),
        "pos": torch.from_numpy(z["pos"]),
    }


def iter_transition_pairs(graphs_root, run_ids: list[str]) -> Iterator[dict[str, torch.Tensor]]:
    """run_ids 각각의 그래프 디렉터리(g000.npz, g001.npz, ...)에서 연속한 (k, k+1) 쌍을
    순서대로 내놓는다. 입력(t_k)은 항상 저장된 정답 그래프에서 로드(팀포싱)."""
    root = Path(graphs_root)
    for rid in sorted(run_ids):
        files = sorted((root / rid).glob("g*.npz"))
        for a, b in zip(files, files[1:]):
            cur = load_graph(a)
            nxt = load_graph(b)
            yield {
                "run_id": rid,
                "node": cur["node"],
                "edge_index": cur["edge_index"],
                "edge": cur["edge"],
                "pos": cur["pos"],
                "true_pos_next": nxt["pos"],
            }


def _norm_to_device(norm: NodeEdgeNorm | None, device: str) -> NodeEdgeNorm | None:
    if norm is None:
        return None
    return NodeEdgeNorm(
        node_mean=norm.node_mean.to(device),
        node_std=norm.node_std.to(device),
        edge_mean=norm.edge_mean.to(device),
        edge_std=norm.edge_std.to(device),
    )


def _prepare_pair(
    pair: dict[str, torch.Tensor], norm_dev: NodeEdgeNorm | None, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node = pair["node"].to(device)
    edge_index = pair["edge_index"].to(device)
    edge = pair["edge"].to(device)
    pos = pair["pos"].to(device)
    true_pos_next = pair["true_pos_next"].to(device)
    node_in = normalize_node(node, norm_dev) if norm_dev is not None else node
    edge_in = normalize_edge(edge, norm_dev) if norm_dev is not None else edge
    return node_in, edge_index, edge_in, pos, true_pos_next


def train_one_epoch(
    model: GraphDisplacementModel,
    optimizer: torch.optim.Optimizer,
    graphs_root,
    run_ids: list[str],
    norm: NodeEdgeNorm | None = None,
    device: str = "cpu",
) -> float:
    """전이쌍 전체를 한 번씩 돌며(팀포싱) 학습한다. 쌍마다 개별 역전파(점 개수가 쌍마다
    달라 배치로 묶기 어려움). `norm`을 주면 모델에 넣기 전 node/edge를 정규화(출력 변위는
    정규화하지 않음 — 체크포인트에 별도 출력 통계가 없어 raw 그대로 사용). `device`("cpu"/
    "cuda")로 각 텐서를 옮긴다 — 모델을 해당 device로 옮기는 건 호출자 책임(W4-04,
    `train_main.run_training`이 루프 시작 전 한 번 `model.to(device)`). "cpu"는 기존과
    동일(무연산 `.to`)해 하위 호환. 평균 손실을 반환."""
    total, n = 0.0, 0
    norm_dev = _norm_to_device(norm, device)
    for pair in iter_transition_pairs(graphs_root, run_ids):
        optimizer.zero_grad()
        node_in, edge_index, edge_in, pos, true_pos_next = _prepare_pair(pair, norm_dev, device)
        disp = model(node_in, edge_index, edge_in)
        pred_pos = pos + disp
        loss = chamfer_loss(pred_pos, true_pos_next)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: GraphDisplacementModel,
    graphs_root,
    run_ids: list[str],
    norm: NodeEdgeNorm | None = None,
    device: str = "cpu",
) -> float:
    """`train_one_epoch`과 같은 teacher-forcing 손실을 역전파 없이 계산한다(W4-04,
    `train_main.run_training`이 매 에폭 val 체크포인트 선택에 씀). `model.eval()`로 전환하고
    끝나면 원래 모드(`training`/`eval`)로 되돌린다."""
    was_training = model.training
    model.eval()
    norm_dev = _norm_to_device(norm, device)
    total, n = 0.0, 0
    for pair in iter_transition_pairs(graphs_root, run_ids):
        node_in, edge_index, edge_in, pos, true_pos_next = _prepare_pair(pair, norm_dev, device)
        disp = model(node_in, edge_index, edge_in)
        pred_pos = pos + disp
        loss = chamfer_loss(pred_pos, true_pos_next)
        total += float(loss)
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)
