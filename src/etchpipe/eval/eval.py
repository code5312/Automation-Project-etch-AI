"""C4/W5-02: 평가 진입점.

체크포인트 + `dataset_hash`(설정 파일을 통해)를 받아 val 전체에 대해 모델 A 롤아웃 지표를
스텝 5·10·20(및 최종 스텝)별로 계산하고 `metrics.json`·`plots/<run_id>.png`(예측 vs 정답
오버레이)를 쓴다. 결정적 실행을 위해 시드를 고정하고, 재현에 필요한 하드웨어·시드·git
SHA·패키지 버전을 `metrics.json`에 같이 남긴다(`run_train.py`의 W4-01 장치를 그대로 재사용).

모델 B(FNO)는 아직 이 repo에 없어 이 진입점은 모델 A만 다룬다(W6-11 이후 확장 예정).
"""

import argparse
import json
import pathlib
import platform
import subprocess

import numpy as np
import torch
import yaml

from etchpipe.train.loop import GraphDisplacementModel, NodeEdgeNorm, load_graph, load_norm_stats
from etchpipe.train.rollout import rollout, rollout_val_error_by_step
from etchpipe.train.run_train import git_sha, package_versions, set_seed

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_STEPS = (5, 10, 20)


def load_model_a(
    ckpt_path: pathlib.Path, device: str = "cpu"
) -> tuple[GraphDisplacementModel, str]:
    """체크포인트의 실제 `processor_layers` 개수로 `processor_size`를 역산해서 로드
    (`scripts/run_rollout_comparison.py`에서 옮김 — physicsnemo MeshGraphNet은
    processor_size개의 edge_block·node_block을 번갈아 쌓아 레이어 개수=2*processor_size,
    W4-06에서 실측으로 확인). `train_main.py`가 저장한 체크포인트의 `net.` 접두사도 지원.
    `MeshGraphNetModel`은 함수 안에서 지연 임포트 — physicsnemo가 없는 호스트에서도
    `eval.py`의 나머지(집계·플롯·CLI)는 토이 모델로 테스트 가능하게 한다(train_main.py의
    `build_model`과 같은 이유)."""
    from etchpipe.models.meshgraphnet import MeshGraphNetModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("net.") for k in sd):
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
    n_layer_entries = 1 + max(
        int(k.split(".")[2]) for k in sd if k.startswith("processor.processor_layers.")
    )
    processor_size = n_layer_entries // 2
    model = MeshGraphNetModel(processor_size=processor_size)
    model.net.load_state_dict(sd)
    model.to(device)
    model.eval()
    note = f"epoch={ckpt.get('epoch', '[미확인]')}, processor_size={processor_size}"
    return model, note


def _agg(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": len(vals)}


def safe_git_sha() -> str:
    """git이 없는 환경(예: etch-train:0.1 컨테이너, `test_run_train.py`의 기존 3건과 같은
    한계)에서도 이미 다 계산한 지표(val 전체 롤아웃 등)를 이 메타데이터 하나 때문에
    버리지 않는다 — 2026-11-10, 실제 체크포인트로 처음 돌려보다 이 순서로 발견."""
    try:
        return git_sha()
    except (OSError, subprocess.CalledProcessError):
        return "[미확인: git 명령 실패]"


def hardware_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _plot_overlay(
    run_id: str, pred: np.ndarray, true: np.ndarray, chamfer: float, out_path: pathlib.Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # 디스플레이 없는 CI/컨테이너에서도 저장만 하면 됨
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(true[:, 0], true[:, 1], s=4, label="true", alpha=0.6)
    ax.scatter(pred[:, 0], pred[:, 1], s=4, label="pred", alpha=0.6)
    ax.set_title(f"{run_id} (chamfer={chamfer:.3f})")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_eval(
    model: GraphDisplacementModel,
    graphs_root: pathlib.Path,
    raw_dir: pathlib.Path,
    val_ids: list[str],
    norm: NodeEdgeNorm | None,
    out_dir: pathlib.Path,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    n_plots: int = 5,
) -> dict:
    """val 전체(정렬된 순서로 — 결정적)에 대해 스텝 5·10·20·최종 지표를 계산하고,
    앞 `n_plots`개 run에 대해 최종 스텝 예측 vs 정답 오버레이 PNG를 남긴다."""
    out_dir = pathlib.Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    by_step: dict[int, list[float]] = {s: [] for s in steps}
    by_step_depth: dict[int, list[float]] = {s: [] for s in steps}
    final_chamfer, final_depth = [], []

    for i, rid in enumerate(sorted(val_ids)):
        meta = json.loads((raw_dir / rid / "meta.json").read_text())
        actual_k = meta["actual_k"]
        run_steps = sorted({s for s in steps if s <= actual_k} | {actual_k})
        result = rollout_val_error_by_step(
            model, graphs_root, rid, run_steps, meta["step_min"], norm=norm
        )
        entry = {"run_id": rid, "actual_k": actual_k, "steps": result}
        per_run.append(entry)
        for s in steps:
            if s in result:
                by_step[s].append(result[s]["chamfer"])
                by_step_depth[s].append(result[s]["depth_error"])
        final_chamfer.append(result[actual_k]["chamfer"])
        final_depth.append(result[actual_k]["depth_error"])

        if i < n_plots:
            g0 = load_graph(pathlib.Path(graphs_root) / rid / "g000.npz")
            positions = rollout(
                model, g0["node"], g0["edge_index"], g0["edge"], g0["pos"],
                actual_k, meta["step_min"], norm,
            )
            pred_final = positions[-1].numpy()
            true_final = load_graph(
                pathlib.Path(graphs_root) / rid / f"g{actual_k:03d}.npz"
            )["pos"].numpy()
            _plot_overlay(
                rid, pred_final, true_final, result[actual_k]["chamfer"],
                plots_dir / f"{rid}.png",
            )

    return {
        "n_val": len(val_ids),
        "by_step": {
            str(s): {"chamfer": _agg(by_step[s]), "depth_error": _agg(by_step_depth[s])}
            for s in steps
            if by_step[s]
        },
        "final_step": {"chamfer": _agg(final_chamfer), "depth_error": _agg(final_depth)},
        "per_run": per_run,
    }


def main(
    checkpoint_path: str | pathlib.Path,
    config_path: str | pathlib.Path = REPO_ROOT / "configs" / "model" / "mgn_v1.yaml",
    out_dir: str | pathlib.Path = REPO_ROOT / "reports" / "eval",
    device: str = "cpu",
    steps: tuple[int, ...] = DEFAULT_STEPS,
    n_plots: int = 5,
    seed: int = 0,
) -> pathlib.Path:
    set_seed(seed)
    config = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
    dataset_hash = config["data"]["dataset_hash"]
    data_dir = REPO_ROOT / "data"
    graphs_root = data_dir / "processed" / dataset_hash / "graphs"
    raw_dir = data_dir / "raw"
    norm = load_norm_stats(data_dir / "processed" / dataset_hash / "stats.json")
    splits = json.loads((data_dir / "splits.json").read_text())

    model, checkpoint_note = load_model_a(pathlib.Path(checkpoint_path), device=device)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        model, graphs_root, raw_dir, splits["val"], norm, out_dir, steps=steps, n_plots=n_plots
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_note"] = checkpoint_note
    metrics["dataset_hash"] = dataset_hash
    metrics["seed"] = seed
    metrics["git_sha"] = safe_git_sha()
    metrics["packages"] = package_versions()
    metrics["hardware"] = hardware_info(device)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"n_val={metrics['n_val']}")
    print(f"final_step={json.dumps(metrics['final_step'], indent=2)}")
    print(f"-> {metrics_path}")
    return metrics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_path")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "model" / "mgn_v1.yaml"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "eval"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--steps", default="5,10,20")
    parser.add_argument("--n-plots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        args.checkpoint_path,
        args.config,
        args.out_dir,
        args.device,
        tuple(int(s) for s in args.steps.split(",")),
        args.n_plots,
        args.seed,
    )
