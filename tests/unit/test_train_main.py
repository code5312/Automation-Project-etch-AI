"""W4-04: 본 학습 진입점(train_main.run_training) 단위 테스트.

실제 MeshGraphNet(physicsnemo 의존)은 호스트에 없을 수 있어 토이 모델로 루프 배선만
검증한다(에폭마다 metrics.json·ckpt_last.pt 갱신, 시간 예산 초과 시 중단, val 손실
추적·`ckpt_best.pt` 저장). 실제 모델 연결은 W4-03/04 증거(컨테이너 실행)로 이미 확인됨.
"""

import json
import pathlib

import pytest
import torch
from torch import nn

import etchpipe.train.train_main as train_main
from etchpipe.etl.etl import THETA_KEYS, etl_run
from etchpipe.train.train_main import run_training

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "raw"
RUN_IDS = ["r0000", "r0001", "r0002"]


class ToyNodeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(14, 2)

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


def test_run_training_writes_metrics_and_checkpoint_each_epoch(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.manual_seed(0)
    model = ToyNodeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    metrics = run_training(model, opt, run_dir, graphs_root, RUN_IDS, epochs=3)

    assert len(metrics) == 3
    assert all("loss" in m and "epoch_time_s" in m for m in metrics)
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved == metrics
    ckpt = torch.load(run_dir / "ckpt_last.pt", weights_only=True)
    assert ckpt["epoch"] == 2
    assert "state_dict" in ckpt


def test_run_training_stops_immediately_when_time_budget_already_exceeded(
    tmp_path: pathlib.Path,
) -> None:
    graphs_root = _build_graphs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = ToyNodeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    metrics = run_training(
        model, opt, run_dir, graphs_root, RUN_IDS, epochs=5, time_budget_s=-1.0
    )

    assert len(metrics) == 1
    assert metrics[0]["stopped"] == "time_budget_exceeded"
    assert not (run_dir / "ckpt_last.pt").exists()
    assert not (run_dir / "metrics.json").exists()


def test_run_training_records_val_loss_each_epoch(tmp_path: pathlib.Path) -> None:
    graphs_root = _build_graphs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.manual_seed(0)
    model = ToyNodeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    metrics = run_training(
        model,
        opt,
        run_dir,
        graphs_root,
        train_ids=RUN_IDS[:2],
        epochs=3,
        val_ids=RUN_IDS[2:],
    )

    assert all("val_loss" in m for m in metrics)
    assert (run_dir / "ckpt_best.pt").exists()


def test_run_training_saves_best_checkpoint_by_val_loss_not_last_epoch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = ToyNodeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    val_losses = iter([1.0, 0.5, 0.8])
    monkeypatch.setattr(train_main, "train_one_epoch", lambda *a, **k: 0.0)
    monkeypatch.setattr(train_main, "evaluate", lambda *a, **k: next(val_losses))

    metrics = run_training(
        model, opt, run_dir, graphs_root=tmp_path, train_ids=[], epochs=3, val_ids=["dummy"]
    )

    assert [m["val_loss"] for m in metrics] == [1.0, 0.5, 0.8]
    best = torch.load(run_dir / "ckpt_best.pt", weights_only=True)
    assert best["epoch"] == 1
    assert best["val_loss"] == 0.5
