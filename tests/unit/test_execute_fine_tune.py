"""ADR-0006/W9-03: `retrain/execute.py::make_fine_tune_fn`의 베스트 체크포인트 선택
단위 테스트.

실제 MeshGraphNet(`load_model_a`)·`train_one_epoch`/`evaluate`는 전부 더미로 바꿔
오케스트레이션 로직(매 에폭 검증 손실을 재서 가장 낮은 에폭의 state_dict를 저장하는가)
만 검증한다 — physicsnemo/torch_scatter 없이도 돈다. 실제 모델로의 통합 동작(진짜
MeshGraphNet·진짜 드리프트 데이터)은 `tests/unit/test_execute.py`의 방침과 같게
`reports/evidence/`의 실행 증거로 검증(로컬에 physicsnemo 없음, 2026-11-24 확인)."""

import json
import pathlib
import types

import pytest
import torch
from torch import nn

import etchpipe.eval.eval as eval_mod
import etchpipe.train.loop as loop_mod
from etchpipe.retrain.execute import make_fine_tune_fn
from etchpipe.train.loop import NodeEdgeNorm


class _FakeModel(nn.Module):
    """`epoch_marker`만 들고 있는 더미 모델 — `state_dict()`가 "어느 에폭 직후
    캡처됐는지"를 그대로 드러내도록만 설계(실제 가중치가 아님)."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.epoch_marker = -1

    def state_dict(self, *args, **kwargs):  # noqa: D102 - 테스트용 오버라이드
        return {"marker": torch.tensor([float(self.epoch_marker)])}


def _fake_norm() -> NodeEdgeNorm:
    return NodeEdgeNorm(
        node_mean=torch.zeros(1),
        node_std=torch.ones(1),
        edge_mean=torch.zeros(1),
        edge_std=torch.ones(1),
    )


def _patch_training_fns(
    monkeypatch: pytest.MonkeyPatch, model: _FakeModel, val_losses: list[float]
) -> None:
    def fake_train_one_epoch(
        model, optimizer, graphs_root, run_ids, norm=None, device="cpu", cache=None
    ):
        model.epoch_marker += 1
        return 0.0

    def fake_evaluate(model, graphs_root, run_ids, norm=None, device="cpu", cache=None):
        return val_losses[model.epoch_marker]

    def fake_load_model_a(ckpt_path):
        return model, "fake", None

    monkeypatch.setattr(eval_mod, "load_model_a", fake_load_model_a)
    monkeypatch.setattr(loop_mod, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(loop_mod, "evaluate", fake_evaluate)


def test_fine_tune_fn_saves_best_val_epoch_not_last(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val_losses = [0.5, 0.3, 0.6, 0.4]  # 최저는 epoch 1(인덱스) — 마지막(epoch 3)이 아님
    model = _FakeModel()
    _patch_training_fns(monkeypatch, model, val_losses)

    state = types.SimpleNamespace(
        graphs_root=tmp_path, splits={"train": ["dummy_train"], "val": ["dummy_val"]}
    )
    fine_tune_fn = make_fine_tune_fn(state, _fake_norm(), out_root=tmp_path)

    ckpt_path = fine_tune_fn(
        pathlib.Path("fake_champion.pt"), {"attempt": 1, "epochs": (4, 4), "lr_mult": 1.0}
    )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 1
    assert saved["state_dict"]["marker"].item() == 1.0

    metrics = json.loads((ckpt_path.parent / "metrics.json").read_text())
    assert metrics["best_epoch"] == 1
    assert metrics["best_val_loss"] == pytest.approx(0.3)
    assert len(metrics["epochs"]) == 4
    assert metrics["epochs"][1]["val_loss"] == pytest.approx(0.3)


def test_fine_tune_fn_falls_back_to_last_epoch_when_no_val(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _FakeModel()
    _patch_training_fns(monkeypatch, model, val_losses=[])

    state = types.SimpleNamespace(
        graphs_root=tmp_path, splits={"train": ["dummy_train"], "val": []}
    )
    fine_tune_fn = make_fine_tune_fn(state, _fake_norm(), out_root=tmp_path)

    ckpt_path = fine_tune_fn(
        pathlib.Path("fake_champion.pt"), {"attempt": 1, "epochs": (3, 3), "lr_mult": 1.0}
    )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 2  # 마지막 에폭(인덱스 2) 그대로

    metrics = json.loads((ckpt_path.parent / "metrics.json").read_text())
    assert metrics["best_val_loss"] is None
    assert "val_loss" not in metrics["epochs"][0]
