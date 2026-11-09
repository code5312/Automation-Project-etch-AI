"""W4-02: 학습 루프(전이쌍, 팀포싱) 단위 테스트.

tests/fixtures/raw(r0000~r0002, K=3, 실제 ViennaPS 결과)를 etl_run으로 그래프로 변환한 뒤,
토이 모델(노드 특징만 보는 Linear)로 루프 플러밍·손실 감소를 확인한다. 실제 MeshGraphNet
연결·과적합 확인은 W4-03.
"""

import pathlib

import torch
from torch import nn

from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.train.loop import GraphDisplacementModel, iter_transition_pairs, train_one_epoch

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"
RUN_IDS = ["r0000", "r0001", "r0002"]


class ToyNodeMLP(nn.Module):
    """edge_index/edge는 무시하는 가장 단순한 모델 — 루프 플러밍 검증용(실제 모델 아님)."""

    def __init__(self, n_in: int = 14, n_out: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        del edge_index, edge
        return self.lin(node)


def _build_graphs(tmp_path: pathlib.Path) -> pathlib.Path:
    graphs_root = tmp_path / "graphs"
    for rid in RUN_IDS:
        etl_run(FIXTURES / rid, graphs_root / rid, THETA_KEYS)
    return graphs_root


def test_iter_transition_pairs_counts_k_pairs_per_run(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    pairs = list(iter_transition_pairs(graphs_root, RUN_IDS))
    # 각 run은 actual_k=3 -> 스냅샷 4개(t000~t003) -> 연속 쌍 3개, run 3개 x 3 = 9
    assert len(pairs) == 9
    assert {p["run_id"] for p in pairs} == set(RUN_IDS)


def test_iter_transition_pairs_shapes(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    pair = next(iter_transition_pairs(graphs_root, RUN_IDS))
    n = pair["node"].shape[0]
    assert pair["node"].shape == (n, 14)
    assert pair["pos"].shape == (n, 2)
    assert pair["edge_index"].shape[0] == 2
    assert pair["edge"].shape[1] == 3
    assert pair["true_pos_next"].shape[1] == 2


def test_train_one_epoch_reduces_loss_over_epochs(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    torch.manual_seed(0)
    model: GraphDisplacementModel = ToyNodeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = [train_one_epoch(model, opt, graphs_root, RUN_IDS) for _ in range(20)]

    assert losses[-1] < losses[0]


def test_train_one_epoch_updates_model_parameters(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    torch.manual_seed(0)
    model: GraphDisplacementModel = ToyNodeMLP()
    before = model.lin.weight.clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    train_one_epoch(model, opt, graphs_root, RUN_IDS)

    assert not torch.equal(before, model.lin.weight)
