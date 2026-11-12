"""W4-04: 본 학습 진입점.

`configs/model/*.yaml` + `data/splits.json` + `data/processed/<dataset_hash>/`를 엮어
실제 학습을 돈다. 배선(이 파일)은 [CPU/매핑] 담당이 작성·검증(토이 모델, 실제 데이터
fixture)하고, `device="cuda"`로 긴 시간 실제 실행하는 쪽은 GPU 보유자가 맡는다(2026-11-09
역할분담, docs/project-checklist.md).

`build_model`에서만 `etchpipe.models.meshgraphnet`(physicsnemo 의존)을 지연 임포트해서,
`run_training` 자체는 physicsnemo가 없는 호스트에서도 토이 모델로 테스트 가능하게 한다.
"""

import argparse
import json
import pathlib
import time

import torch
import yaml

from etchpipe.train.loop import (
    GraphDisplacementModel,
    NodeEdgeNorm,
    evaluate,
    load_norm_stats,
    train_one_epoch,
)
from etchpipe.train.run_train import init_run

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def run_training(
    model: GraphDisplacementModel,
    optimizer: torch.optim.Optimizer,
    run_dir: pathlib.Path,
    graphs_root,
    train_ids: list[str],
    epochs: int,
    norm: NodeEdgeNorm | None = None,
    device: str = "cpu",
    time_budget_s: float | None = None,
    val_ids: list[str] | None = None,
    val_every: int = 1,
) -> list[dict]:
    """에폭 루프. 매 에폭 `ckpt_last.pt`(state_dict+epoch)와 `metrics.json`(누적)을 덮어써서,
    중간에 멈춰도(시간 예산 초과·수동 중단) 마지막 에폭까지의 결과가 남는다. `time_budget_s`를
    넘기면 그 시점에서 멈추고 `{"stopped": "time_budget_exceeded", ...}` 기록만 추가한다
    (체크포인트는 그 직전 에폭 것 그대로).

    `val_ids`를 주면 `val_every`에폭마다(기본 매 에폭) `evaluate`로 val 손실을 계산해
    `metrics`에 `val_loss`로 남기고, 지금까지 중 최소면 `ckpt_best.pt`(state_dict+epoch+
    val_loss)를 따로 저장한다 — `ckpt_last.pt`는 항상 마지막 에폭 그대로, 과적합으로
    마지막 에폭이 최선이 아닐 수 있다는 문제(2026-11-10,
    `reports/model_issues_mgn.md` 문제①)를 고치는 목적."""
    model.to(device)
    run_dir = pathlib.Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "ckpt_last.pt"
    ckpt_best_path = run_dir / "ckpt_best.pt"
    metrics: list[dict] = []
    best_val_loss: float | None = None
    t_start = time.time()
    for epoch in range(epochs):
        elapsed = time.time() - t_start
        if time_budget_s is not None and elapsed > time_budget_s:
            metrics.append(
                {"epoch": epoch, "stopped": "time_budget_exceeded", "elapsed_s": elapsed}
            )
            break
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, graphs_root, train_ids, norm=norm, device=device)
        epoch_time_s = time.time() - t0
        entry = {"epoch": epoch, "loss": loss, "epoch_time_s": epoch_time_s}
        if val_ids is not None and epoch % val_every == 0:
            val_loss = evaluate(model, graphs_root, val_ids, norm=norm, device=device)
            entry["val_loss"] = val_loss
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                    ckpt_best_path,
                )
        metrics.append(entry)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_model(model_config: dict) -> GraphDisplacementModel:
    name = model_config["name"]
    if name == "meshgraphnet":
        from etchpipe.models.meshgraphnet import MeshGraphNetModel

        return MeshGraphNetModel(**model_config["meshgraphnet"])
    raise ValueError(f"unknown model.name: {name!r}")


def main(
    config_path: str | pathlib.Path,
    out_root: str | pathlib.Path = "runs",
    device: str = "cpu",
    time_budget_hours: float | None = None,
    splits_path: str | pathlib.Path | None = None,
    val_every: int = 1,
) -> pathlib.Path:
    config = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
    run_dir = init_run(config_path, out_root=out_root)

    dataset_hash = config["data"]["dataset_hash"]
    graphs_root = REPO_ROOT / "data" / "processed" / dataset_hash / "graphs"
    stats_path = REPO_ROOT / "data" / "processed" / dataset_hash / "stats.json"
    splits_path = pathlib.Path(splits_path) if splits_path else REPO_ROOT / "data" / "splits.json"

    splits = json.loads(splits_path.read_text())
    train_ids = splits["train"]
    val_ids = splits["val"]
    norm = load_norm_stats(stats_path)

    model = build_model(config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    time_budget_s = time_budget_hours * 3600 if time_budget_hours is not None else None
    metrics = run_training(
        model,
        optimizer,
        run_dir,
        graphs_root,
        train_ids,
        epochs=config["train"]["epochs"],
        norm=norm,
        device=device,
        time_budget_s=time_budget_s,
        val_ids=val_ids,
        val_every=val_every,
    )
    n_done = sum(1 for m in metrics if "loss" in m)
    print(f"run_dir={run_dir}")
    print(f"epochs_completed={n_done}/{config['train']['epochs']}")
    if metrics and "loss" in metrics[-1]:
        print(f"last_loss={metrics[-1]['loss']:.4f}")
        if "val_loss" in metrics[-1]:
            print(f"last_val_loss={metrics[-1]['val_loss']:.4f}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--splits-path", default=None)
    parser.add_argument(
        "--val-every", type=int, default=1, help="val 평가를 몇 에폭마다 돌릴지(기본 매 에폭)"
    )
    args = parser.parse_args()
    main(
        args.config_path,
        args.out_root,
        args.device,
        args.time_budget_hours,
        args.splits_path,
        args.val_every,
    )
