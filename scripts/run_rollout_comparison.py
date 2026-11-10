"""W4-06: 모델 A 롤아웃 오차 vs 베이스라인(a)(b) 비교표.

베이스라인 두 종(W3-07, `reports/evidence/W3-07_baseline_metrics.json`)과 같은 틀
(val 40건, t=0 -> t=actual_k 최종 표면)로 모델 A의 **롤아웃**(W4-05, `train/rollout.py`)
오차를 비교한다. torch/physicsnemo가 필요해 train 컨테이너(`etch-train:0.1`) 안에서만
돈다(베이스라인 쪽 `run_baselines.py`는 반대로 torch 없이 호스트에서 돈다).

사용: PYTHONPATH=src python scripts/run_rollout_comparison.py [checkpoint.pt]
  checkpoint 생략 시 runs/mgn_draft/ckpt_mgn.pt를 기본으로 찾는다. 체크포인트가 없으면
  모델 A 행은 [미확인]으로 비워 두고 베이스라인만 기록한다(스크립트 자체는 그래도 끝까지
  돈다 — W4-04 본 학습 체크포인트가 나오면 그 경로를 인자로 넘겨서 재실행).
출력: reports/evidence/W4-06_rollout_vs_baseline.json
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import yaml

from etchpipe.models.meshgraphnet import MeshGraphNetModel
from etchpipe.train.loop import load_norm_stats
from etchpipe.train.rollout import rollout_val_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "runs" / "mgn_draft" / "ckpt_mgn.pt"


def _agg(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}


def _load_model_a(ckpt_path: pathlib.Path) -> tuple[MeshGraphNetModel, str]:
    """체크포인트를 읽어 state_dict의 실제 processor_layers 개수로 processor_size를
    역산한다. physicsnemo MeshGraphNet은 processor_size개의 edge_block과 processor_size개의
    node_block을 번갈아 쌓으므로 processor_layers 개수 = 2*processor_size
    (2026-11-09, load_state_dict 실패로 발견·정정 — 레이어 개수를 그대로 processor_size로
    착각했던 첫 시도는 틀렸음)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # train_main.py(W4-04)로 직접 학습한 체크포인트는 MeshGraphNetModel 래퍼 전체를
    # model.state_dict()로 저장해서 키 앞에 "net."이 붙는다(팀원 원본 체크포인트는 접두사
    # 없음) — 2026-11-10, 실제 체크포인트로 처음 로드해보다 발견. 둘 다 지원하도록 벗겨낸다.
    if any(k.startswith("net.") for k in sd):
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
    n_layer_entries = 1 + max(
        int(k.split(".")[2])
        for k in sd
        if k.startswith("processor.processor_layers.")
    )
    processor_size = n_layer_entries // 2
    model = MeshGraphNetModel(processor_size=processor_size)
    model.net.load_state_dict(sd)
    model.eval()
    note = f"epoch={ckpt.get('epoch', '[미확인]')}, processor_size={processor_size}"
    return model, note


def main() -> int:
    data_dir = ROOT / "data"
    splits = json.loads((data_dir / "splits.json").read_text())
    config = yaml.safe_load((ROOT / "configs" / "model" / "mgn_v1.yaml").read_text())
    dataset_hash = config["data"]["dataset_hash"]
    graphs_root = data_dir / "processed" / dataset_hash / "graphs"
    norm = load_norm_stats(data_dir / "processed" / dataset_hash / "stats.json")

    baseline_path = ROOT / "reports" / "evidence" / "W3-07_baseline_metrics.json"
    baseline = json.loads(baseline_path.read_text())

    ckpt_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    model_a_section: dict = {"checkpoint": str(ckpt_path)}
    if ckpt_path.exists():
        model, note = _load_model_a(ckpt_path)
        model_a_section["checkpoint_note"] = note
        per_run = []
        for rid in splits["val"]:
            meta = json.loads((data_dir / "raw" / rid / "meta.json").read_text())
            err = rollout_val_error(
                model,
                graphs_root,
                rid,
                n_steps=meta["actual_k"],
                step_min=meta["step_min"],
                norm=norm,
            )
            per_run.append({"run_id": rid, **err})
        model_a_section["chamfer"] = _agg([r["chamfer"] for r in per_run])
        model_a_section["depth_error"] = _agg([r["depth_error"] for r in per_run])
        model_a_section["per_run"] = per_run
    else:
        model_a_section["note"] = "[미확인] 체크포인트 없음 — W4-04 본 학습 완료 후 재실행"

    summary = {
        "n_val": len(splits["val"]),
        "model_a_rollout": model_a_section,
        "a_zero_displacement": baseline["a_zero_displacement"],
        "b_depth_mlp_extrapolation": baseline["b_depth_mlp_extrapolation"],
    }

    out_path = ROOT / "reports" / "evidence" / "W4-06_rollout_vs_baseline.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "model_a_rollout"}, indent=2))
    if "chamfer" in model_a_section:
        print(
            "model_a_rollout:",
            json.dumps(
                {k: v for k, v in model_a_section.items() if k != "per_run"}, indent=2
            ),
        )
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
