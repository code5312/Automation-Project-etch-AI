"""C3: 학습 루프(W4-02) — 전이쌍(t_k -> t_{k+1}) + 팀포싱.

"팀포싱": 각 스텝의 입력은 항상 **정답** 그래프(t_k)에서 가져온다(모델 자신의 이전 예측을
다시 입력으로 쓰는 자기회귀 롤아웃이 아니다) — 평가 단계의 다단계 롤아웃(W4-05)과는 다른
학습 방식. 모델은 입력 그래프의 각 점에 대한 변위(dx,dy)를 예측하고, `pos+변위`를
`chamfer_loss`로 다음 스냅샷(t_{k+1})의 점들과 비교한다(점 개수가 달라도 됨).

모델 인터페이스: `model(node, edge_index, edge) -> displacement`
  node: (N,14) float32, edge_index: (2,2E) int64, edge: (2E,3) float32, 반환: (N,2) float32.
실제 MeshGraphNet(nvidia-physicsnemo) 연결 시 이 시그니처가 맞지 않으면 W4-03에서 얇은
래퍼로 맞춘다[미확인: physicsnemo MeshGraphNet.forward의 정확한 입력 형식].
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from etchpipe.train.chamfer_loss import chamfer_loss


class GraphDisplacementModel(Protocol):
    def __call__(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor: ...


def _load_graph(path) -> dict[str, torch.Tensor]:
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
            cur = _load_graph(a)
            nxt = _load_graph(b)
            yield {
                "run_id": rid,
                "node": cur["node"],
                "edge_index": cur["edge_index"],
                "edge": cur["edge"],
                "pos": cur["pos"],
                "true_pos_next": nxt["pos"],
            }


def train_one_epoch(
    model: GraphDisplacementModel,
    optimizer: torch.optim.Optimizer,
    graphs_root,
    run_ids: list[str],
) -> float:
    """전이쌍 전체를 한 번씩 돌며(팀포싱) 학습한다. 쌍마다 개별 역전파(점 개수가 쌍마다
    달라 배치로 묶기 어려움). 평균 손실을 반환."""
    total, n = 0.0, 0
    for pair in iter_transition_pairs(graphs_root, run_ids):
        optimizer.zero_grad()
        disp = model(pair["node"], pair["edge_index"], pair["edge"])
        pred_pos = pair["pos"] + disp
        loss = chamfer_loss(pred_pos, pair["true_pos_next"])
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)
