"""W4-02: 학습 루프(전이쌍, 팀포싱) 단위 테스트.

tests/fixtures/raw(r0000~r0002, K=3, 실제 ViennaPS 결과)를 etl_run으로 그래프로 변환한 뒤,
토이 모델(노드 특징만 보는 Linear)로 루프 플러밍·손실 감소를 확인한다. 실제 MeshGraphNet
연결·과적합 확인은 W4-03.
"""

import json
import pathlib

import torch
from torch import nn

from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.train.loop import (
    GraphDisplacementModel,
    iter_transition_pairs,
    load_graph,
    load_norm_stats,
    normalize_edge,
    normalize_node,
    train_one_epoch,
)

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


def _write_stats(path: pathlib.Path) -> pathlib.Path:
    stats = {
        "node_mean": [0.0] * 14,
        "node_std": [1.0, 2.0] + [1.0] * 12,
        "edge_mean": [0.0, 0.0, 0.3],
        "edge_std": [0.2, 0.2, 0.1],
    }
    stats_path = path / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return stats_path


def test_load_norm_stats_reads_json(tmp_path: pathlib.Path) -> None:
    stats_path = _write_stats(tmp_path)
    norm = load_norm_stats(stats_path)
    assert norm.node_mean.shape == (14,)
    assert norm.node_std.shape == (14,)
    assert norm.edge_mean.shape == (3,)
    assert norm.edge_std.shape == (3,)
    assert float(norm.node_std[1]) == 2.0


def test_normalize_node_and_edge_apply_mean_std(tmp_path: pathlib.Path) -> None:
    norm = load_norm_stats(_write_stats(tmp_path))
    node = torch.zeros((3, 14))
    node[:, 1] = 4.0
    edge = torch.tensor([[0.0, 0.0, 0.3]])

    node_out = normalize_node(node, norm)
    edge_out = normalize_edge(edge, norm)

    assert torch.allclose(node_out[:, 1], torch.full((3,), 2.0))  # (4-0)/2
    assert torch.allclose(edge_out, torch.zeros_like(edge_out))  # (0.3-0.3)/0.1 등


# W7-07: 전체 파이프라인 실측(증거: reports/evidence/W7-07_pipeline_timing.md)에서
# fine-tune 단계가 전체의 92%를 차지 — 같은 run_ids를 여러 에폭에 걸쳐 돌 때 npz를
# 매번 재읽던 게 그 안의 낭비 중 하나였다. cache는 디스크 재읽기만 생략하고 반환값은
# 동일해야 한다(아래 세 테스트로 확인).
def test_load_graph_with_cache_avoids_second_disk_read(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    path = sorted((graphs_root / "r0000").glob("g*.npz"))[0]
    cache: dict = {}

    first = load_graph(path, cache=cache)
    path.unlink()  # 캐시가 안 쓰이면 다음 load_graph가 FileNotFoundError로 터져야 함
    second = load_graph(path, cache=cache)

    assert second is first  # 같은 캐시 엔트리(새로 읽지 않음)


def test_iter_transition_pairs_with_cache_matches_without_cache(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)

    without_cache = list(iter_transition_pairs(graphs_root, RUN_IDS))
    with_cache = list(iter_transition_pairs(graphs_root, RUN_IDS, cache={}))

    assert len(with_cache) == len(without_cache)
    for a, b in zip(with_cache, without_cache):
        assert torch.equal(a["node"], b["node"])
        assert torch.equal(a["true_pos_next"], b["true_pos_next"])


def test_train_one_epoch_with_cache_gives_same_loss_as_without(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)

    torch.manual_seed(0)
    model_a: GraphDisplacementModel = ToyNodeMLP()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    loss_no_cache = train_one_epoch(model_a, opt_a, graphs_root, RUN_IDS)

    torch.manual_seed(0)
    model_b: GraphDisplacementModel = ToyNodeMLP()
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    loss_with_cache = train_one_epoch(model_b, opt_b, graphs_root, RUN_IDS, cache={})

    assert loss_with_cache == loss_no_cache
    assert torch.equal(model_a.lin.weight, model_b.lin.weight)


def test_train_one_epoch_with_norm_runs_and_updates(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    norm = load_norm_stats(_write_stats(tmp_path))
    torch.manual_seed(0)
    model: GraphDisplacementModel = ToyNodeMLP()
    before = model.lin.weight.clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = train_one_epoch(model, opt, graphs_root, RUN_IDS, norm=norm)

    assert loss >= 0.0
    assert not torch.equal(before, model.lin.weight)
