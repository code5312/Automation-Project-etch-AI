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

# W7-04(보안): 체크포인트 로드를 `weights_only=True`로 바꾼다(기본 `False`는 임의 코드
# 실행이 가능한 일반 pickle — 악성 .pt가 섞여 들어오면 위험). 체크포인트에 든 mu_n/sd_n/
# mu_e/sd_e가 (구버전 포맷에서는) numpy 배열이라 기본 allowlist만으로는 못 읽어서
# [확인, 2026-11-14 실측: `runs/mgn_tuned/model.pt`로 직접 재현] 필요한 numpy 타입만
# 명시적으로 허용한다 — `weights_only=False`로 되돌리는 것보다 안전하다(그 경로로 들어올
# 수 있는 임의 객체를 이 4개로 좁힘).
torch.serialization.add_safe_globals(
    [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType]
)


def norm_from_checkpoint(ckpt: dict) -> NodeEdgeNorm | None:
    """체크포인트에 내장된 정규화 통계(`mu_n/sd_n/mu_e/sd_e`)가 있으면 그걸로 `NodeEdgeNorm`을
    만든다 [확인, 2026-11-11]. `mgn_tuned`는 이 통계가 `data/processed/<hash>/stats.json`과
    다르다(인덱스 1이 약 15~19% 차이 — 미세조정 중 재계산됐거나 다른 데이터 부분집합 기준으로
    보임) — 외부 `stats.json`을 쓰면 모델이 학습 때와 다른 입력 분포를 보게 돼 틀린다. 없으면
    `None`(호출부가 `load_norm_stats`로 외부 `stats.json`을 대체 사용)."""
    if "mu_n" not in ckpt:
        return None
    return NodeEdgeNorm(
        node_mean=torch.as_tensor(ckpt["mu_n"], dtype=torch.float32),
        node_std=torch.as_tensor(ckpt["sd_n"], dtype=torch.float32),
        edge_mean=torch.as_tensor(ckpt["mu_e"], dtype=torch.float32),
        edge_std=torch.as_tensor(ckpt["sd_e"], dtype=torch.float32),
    )


def load_model_a(
    ckpt_path: pathlib.Path, device: str = "cpu"
) -> tuple[GraphDisplacementModel, str, NodeEdgeNorm | None]:
    """체크포인트의 실제 가중치 shape로 `processor_size`/`hidden_dim_processor`를 역산해서
    로드(`scripts/run_rollout_comparison.py`에서 옮김 — physicsnemo MeshGraphNet은
    processor_size개의 edge_block·node_block을 번갈아 쌓아 레이어 개수=2*processor_size,
    W4-06에서 실측으로 확인). `hidden_dim_processor`는 `node_encoder.model.0.weight`의
    `out_features`에서 읽는다[확인, 2026-11-11 — `runs/mgn_draft`(64)·`runs/mgn_tuned`(128)
    둘 다 이 키로 구분됨, `configs/model/mgn_v1.yaml`의 고정값 하나로는 두 체크포인트를
    동시에 못 맞춰서 자동판별로 바꿈]. `train_main.py`가 저장한 체크포인트의 `net.` 접두사도
    지원. 세 번째 반환값은 `norm_from_checkpoint`(체크포인트 내장 통계, 있으면 우선) —
    `None`이면 호출부가 `dataset_hash` 기준 `stats.json`을 써야 한다. `MeshGraphNetModel`은
    함수 안에서 지연 임포트 — physicsnemo가 없는 호스트에서도 `eval.py`의 나머지(집계·플롯·
    CLI)는 토이 모델로 테스트 가능하게 한다(train_main.py의 `build_model`과 같은 이유)."""
    from etchpipe.models.meshgraphnet import MeshGraphNetModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]
    if any(k.startswith("net.") for k in sd):
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
    n_layer_entries = 1 + max(
        int(k.split(".")[2]) for k in sd if k.startswith("processor.processor_layers.")
    )
    processor_size = n_layer_entries // 2
    hidden_dim_processor = sd["node_encoder.model.0.weight"].shape[0]
    model = MeshGraphNetModel(
        processor_size=processor_size, hidden_dim_processor=hidden_dim_processor
    )
    model.net.load_state_dict(sd)
    model.to(device)
    model.eval()
    note = (
        f"epoch={ckpt.get('epoch', '[미확인]')}, processor_size={processor_size}, "
        f"hidden_dim_processor={hidden_dim_processor}"
    )
    return model, note, norm_from_checkpoint(ckpt)


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
    fix_mask: bool = False,
    guard: bool = False,
    guard_shrink: tuple[float, ...] = (0.5, 0.25, 0.0),
    loop_removal: bool = False,
) -> dict:
    """val 전체(정렬된 순서로 — 결정적)에 대해 스텝 5·10·20·최종 지표를 계산하고,
    앞 `n_plots`개 run에 대해 최종 스텝 예측 vs 정답 오버레이 PNG를 남긴다.

    `fix_mask`/`guard`/`guard_shrink`/`loop_removal`은 체크포인트가 자체 추론 방식에
    맞춰 튜닝된 경우에만 켠다(예: `runs/mgn_tuned`, ADR-0004 후속 배선 2026-11-11) —
    기본은 전부 꺼져 있어 기존 체크포인트 평가와 동일하게 동작한다."""
    out_dir = pathlib.Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    by_step: dict[int, list[float]] = {s: [] for s in steps}
    by_step_depth: dict[int, list[float]] = {s: [] for s in steps}
    by_step_self_x: dict[int, list[int]] = {s: [] for s in steps}
    final_chamfer, final_depth, final_self_x = [], [], []

    for i, rid in enumerate(sorted(val_ids)):
        meta = json.loads((raw_dir / rid / "meta.json").read_text())
        actual_k = meta["actual_k"]
        run_steps = sorted({s for s in steps if s <= actual_k} | {actual_k})
        result = rollout_val_error_by_step(
            model, graphs_root, rid, run_steps, meta["step_min"], norm=norm, check_physics=True,
            fix_mask=fix_mask, guard=guard, guard_shrink=guard_shrink, loop_removal=loop_removal,
        )
        entry = {"run_id": rid, "actual_k": actual_k, "steps": result}
        per_run.append(entry)
        for s in steps:
            if s in result:
                by_step[s].append(result[s]["chamfer"])
                by_step_depth[s].append(result[s]["depth_error"])
                by_step_self_x[s].append(result[s]["self_intersections"])
        final_chamfer.append(result[actual_k]["chamfer"])
        final_depth.append(result[actual_k]["depth_error"])
        final_self_x.append(result[actual_k]["self_intersections"])

        if i < n_plots:
            g0 = load_graph(pathlib.Path(graphs_root) / rid / "g000.npz")
            positions = rollout(
                model, g0["node"], g0["edge_index"], g0["edge"], g0["pos"],
                actual_k, meta["step_min"], norm,
                fix_mask=fix_mask, guard=guard, guard_shrink=guard_shrink,
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
            str(s): {
                "chamfer": _agg(by_step[s]),
                "depth_error": _agg(by_step_depth[s]),
                "self_intersections_total": int(sum(by_step_self_x[s])),
            }
            for s in steps
            if by_step[s]
        },
        "final_step": {
            "chamfer": _agg(final_chamfer),
            "depth_error": _agg(final_depth),
            "self_intersections_total": int(sum(final_self_x)),
        },
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
    fix_mask: bool = False,
    guard: bool = False,
    loop_removal: bool = False,
    split: str = "val",
) -> pathlib.Path:
    """`split`(기본 "val")로 `data/splits.json`의 어느 분할을 평가할지 고른다 [확인,
    W5-03 — "ood"를 넘기면 같은 파이프라인으로 OOD 20개를 평가한다. 설계서 §3 C4
    "OOD 오차 비"는 이 함수 자체가 아니라 호출부가 val/ood 두 번 돌려 직접 비교한다."""
    set_seed(seed)
    config = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
    dataset_hash = config["data"]["dataset_hash"]
    data_dir = REPO_ROOT / "data"
    graphs_root = data_dir / "processed" / dataset_hash / "graphs"
    raw_dir = data_dir / "raw"
    splits = json.loads((data_dir / "splits.json").read_text())

    model, checkpoint_note, ckpt_norm = load_model_a(pathlib.Path(checkpoint_path), device=device)
    norm_source = "checkpoint"
    if ckpt_norm is None:
        ckpt_norm = load_norm_stats(data_dir / "processed" / dataset_hash / "stats.json")
        norm_source = "stats.json"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        model, graphs_root, raw_dir, splits[split], ckpt_norm, out_dir, steps=steps,
        n_plots=n_plots, fix_mask=fix_mask, guard=guard, loop_removal=loop_removal,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_note"] = checkpoint_note
    metrics["split"] = split
    metrics["norm_source"] = norm_source
    metrics["inference_options"] = {
        "fix_mask": fix_mask, "guard": guard, "loop_removal": loop_removal,
    }
    metrics["dataset_hash"] = dataset_hash
    metrics["seed"] = seed
    metrics["git_sha"] = safe_git_sha()
    metrics["packages"] = package_versions()
    metrics["hardware"] = hardware_info(device)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"split={split} n={metrics['n_val']}")
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
    parser.add_argument("--fix-mask", action="store_true", help="마스크 노드 변위 0 고정")
    parser.add_argument("--guard", action="store_true", help="교차가드(0.5/0.25/0 배 축소)")
    parser.add_argument("--loop-removal", action="store_true", help="출력 표면선 자기교차 제거")
    parser.add_argument(
        "--split", default="val", choices=["val", "test", "ood"], help="평가할 분할"
    )
    args = parser.parse_args()
    main(
        args.checkpoint_path,
        args.config,
        args.out_dir,
        args.device,
        tuple(int(s) for s in args.steps.split(",")),
        args.n_plots,
        args.seed,
        args.fix_mask,
        args.guard,
        args.loop_removal,
        args.split,
    )
