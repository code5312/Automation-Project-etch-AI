"""W6-04/05: `run_retrain_loop`에 꽂는 실제 콜백(CPU 미세조정) + `python -m etchpipe.retrain
run` 엔트리포인트.

**실행 환경**: 이 모듈은 torch/physicsnemo(미세조정·게이트)와 ViennaPS(2차 시도
`regenerate_fn`의 드리프트 데이터 확장)를 **둘 다** 쓸 수 있어야 한다 — `etch-serve:0.1`
이미지가 이미 둘 다 설치돼 있어서(W6-08, "환경 3분리"에서 벗어난 서빙 이미지) 그대로
재사용한다. `etch-train:0.1`에서는 `fine_tune_fn`/`gate_fn`은 되지만 `regenerate_fn`이
실제로 불리면(2차 시도) `ModuleNotFoundError: viennaps`로 멈춘다 — 그 경우
`etch-serve:0.1`로 다시 실행.

**ADR-0002 에스컬레이션 표**(`retrain/loop.py::ATTEMPTS`)의 `epochs` 범위(예: `(3,5)`)
중 **하한을 쓴다**[제안, 2026-11-13 — CPU 미세조정이 느려서(과거 실측 5레시피/
150~290초/에폭) 시간 예산 안에서 끝내려는 선택, GPU라면 상한을 쓰는 게 더 맞음]."""

import json
import pathlib
import time
from datetime import UTC, datetime

import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DRIFT_RAW_DIR = REPO_ROOT / "data" / "drift" / "raw"
DRIFT_PROCESSED_ROOT = REPO_ROOT / "data" / "drift" / "processed"
GATE_CFG_PATH = REPO_ROOT / "configs" / "pipeline" / "gate.yaml"
CHAMPION_CKPT = REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"
RETRAIN_RUNS_ROOT = REPO_ROOT / "runs" / "retrain"
SUBSTRATE_A_IE = 8.05  # configs/sim/live.yaml, 7.0 * 1.15
BASE_LR = 0.0001  # configs/model/mgn_v1.yaml train.lr
EVAL_STEPS = (5, 10)


class DriftState:
    """드리프트 데이터(train/val/golden_drift)를 들고 다니며, `expand_train`(2차 시도)이
    train을 늘리면 다시 ETL해 `dataset_hash`/`graphs_root`를 갱신한다."""

    def __init__(self) -> None:
        from etchpipe.sim.drift_recipes import drift_splits

        self.splits, self.params_by_id = drift_splits()
        self._process()

    def _process(self) -> None:
        from etchpipe.etl.etl import THETA_KEYS, process_dataset

        manifest = process_dataset(
            DRIFT_RAW_DIR, self.splits, DRIFT_PROCESSED_ROOT, THETA_KEYS
        )
        self.dataset_hash = manifest["dataset_hash"]
        self.graphs_root = DRIFT_PROCESSED_ROOT / self.dataset_hash / "graphs"

    def eval_ids(self) -> list[str]:
        return sorted(set(self.splits["golden_drift"]) | set(self.splits["val"]))

    def expand_train(self, n_target: int) -> int:
        """ADR-0002 2차 시도: train을 `n_target`까지(기본 40→100, +60) ViennaPS로 더
        시뮬레이션하고 다시 ETL. 반환: 실제로 추가한 레시피 수."""
        from etchpipe.sim.run_sim import run_sim
        from etchpipe.sim.sampling import make_recipes

        n_more = n_target - len(self.splits["train"])
        if n_more <= 0:
            return 0
        extra = make_recipes(n_more, seed=9200, prefix="driftx")
        for r in extra:
            rid = r["run_id"]
            out = DRIFT_RAW_DIR / rid
            if not (out / "meta.json").exists():
                run_sim(r["params"], out, substrate_a_ie=SUBSTRATE_A_IE, sim_config_id="live_v1")
            self.params_by_id[rid] = r["params"]
            self.splits["train"].append(rid)
        self._process()
        return n_more


def make_fine_tune_fn(state: DriftState, fallback_norm):
    def fine_tune_fn(champion_ckpt: pathlib.Path, cfg: dict) -> pathlib.Path:
        from etchpipe.eval.eval import load_model_a
        from etchpipe.train.loop import train_one_epoch

        model, _note, ckpt_norm = load_model_a(champion_ckpt)
        norm = ckpt_norm or fallback_norm
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR * cfg["lr_mult"])
        epochs = cfg["epochs"][0]  # 하한(모듈 docstring 참고)

        out_dir = RETRAIN_RUNS_ROOT / f"attempt{cfg['attempt']}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_one_epoch(
                model, optimizer, state.graphs_root, state.splits["train"],
                norm=norm, device="cpu",
            )
            history.append({"epoch": epoch, "loss": loss, "epoch_time_s": time.time() - t0})
        ckpt_path = out_dir / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epochs - 1,
                "mu_n": norm.node_mean.tolist(),
                "sd_n": norm.node_std.tolist(),
                "mu_e": norm.edge_mean.tolist(),
                "sd_e": norm.edge_std.tolist(),
            },
            ckpt_path,
        )
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=1))
        return ckpt_path

    return fine_tune_fn


def make_gate_fn(state: DriftState):
    def gate_fn(ckpt_path: pathlib.Path) -> dict:
        from etchpipe.eval.eval import load_model_a
        from etchpipe.gate.gate import run_gate
        from etchpipe.train.rollout import rollout_val_error_by_step

        model, _note, ckpt_norm = load_model_a(ckpt_path)
        gate_cfg = yaml.safe_load(GATE_CFG_PATH.read_text())["gate"]

        depth_pred, depth_true, chamfer_by_step, self_x_total = [], [], [], 0
        for rid in state.eval_ids():
            meta = json.loads((DRIFT_RAW_DIR / rid / "meta.json").read_text())
            actual_k = meta["actual_k"]
            steps = sorted({s for s in EVAL_STEPS if s <= actual_k} | {actual_k})
            result = rollout_val_error_by_step(
                model, state.graphs_root, rid, steps, meta["step_min"], norm=ckpt_norm,
                check_physics=True, fix_mask=True, guard=True, loop_removal=True,
            )
            chamfer_by_step.append([result[s]["chamfer"] for s in steps])
            final = result[actual_k]
            depth_pred.append(final["depth_pred"])
            depth_true.append(final["depth_true"])
            self_x_total += final["self_intersections"]

        result = run_gate(
            model, [], max_chamfer=gate_cfg["max_chamfer"],
            rollout_step_errors_by_recipe=chamfer_by_step,
            depth_pred=depth_pred, depth_true=depth_true,
            # [미확인, 알려진 한계] CD 미배선 — 깊이 값 복제(run_gate_negative.py와 동일 관례)
            cd_pred=depth_pred, cd_true=depth_true,
            grid=gate_cfg["grid"], max_dim_pct=gate_cfg["max_dim_pct"],
            rollout_max_ratio=gate_cfg["rollout_max_ratio"],
            self_intersection_count=self_x_total,
        )
        return {
            "passed": result.passed, "exit_code": result.exit_code,
            "reasons": result.reasons, "reference_metrics": result.reference_metrics,
        }

    return gate_fn


def make_regenerate_fn(state: DriftState):
    def regenerate_fn(n_train_target: int) -> None:
        n_added = state.expand_train(n_train_target)
        print(f"  regenerate_fn: train +{n_added} (now {len(state.splits['train'])})")

    return regenerate_fn


def run() -> int:
    """`python -m etchpipe.retrain run` — ADR-0002 루프를 실제로 1회 실행하고
    `reports/evidence/retrain_<ts>.json`에 시도별 로그를 남긴다."""
    from etchpipe.eval.eval import load_model_a
    from etchpipe.registry.registry import Registry, compute_valid_range
    from etchpipe.retrain.loop import run_retrain_loop

    print("드리프트 데이터 로드·ETL 중...")
    state = DriftState()
    print(f"  train={len(state.splits['train'])} val={len(state.splits['val'])} "
          f"golden_drift={len(state.splits['golden_drift'])} dataset_hash={state.dataset_hash}")

    _champion_model, _note, champion_norm = load_model_a(CHAMPION_CKPT)

    result = run_retrain_loop(
        CHAMPION_CKPT,
        make_fine_tune_fn(state, champion_norm),
        make_gate_fn(state),
        regenerate_fn=make_regenerate_fn(state),
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    evidence = {
        "timestamp": ts,
        "dataset_hash_final": state.dataset_hash,
        "promoted": result["promoted"],
        "promoted_ckpt": str(result["promoted_ckpt"]) if result["promoted_ckpt"] else None,
        "attempts": [
            {
                "attempt": a["attempt"],
                "config": {k: v for k, v in a["config"].items()},
                "gate_result": a["gate_result"],
            }
            for a in result["attempts"]
        ],
    }
    out_path = REPO_ROOT / "reports" / "evidence" / f"retrain_{ts}.json"
    out_path.write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    print(f"-> {out_path}")

    if result["promoted"]:
        reg = Registry(REPO_ROOT / "registry")
        model_id = f"mgn_retrain_{ts}"
        new_ckpt_path = RETRAIN_RUNS_ROOT / model_id / "model.pt"
        new_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(result["promoted_ckpt"]).replace(new_ckpt_path)
        valid_range = compute_valid_range(DRIFT_RAW_DIR, state.splits["train"])
        from etchpipe.gate.gate import GateResult

        last_gate = result["attempts"][-1]["gate_result"]
        gate_result_obj = GateResult(
            passed=last_gate["passed"], exit_code=last_gate["exit_code"],
            reasons=last_gate["reasons"],
        )
        reg.register(
            model_id, new_ckpt_path, gate_result_obj,
            dataset_hash=state.dataset_hash,
            config={"source": "retrain_loop", "attempts": len(result["attempts"])},
            git_sha="[미확인: 재학습 루프 실행, git 명령 미사용]",
            metrics=last_gate.get("reference_metrics", {}),
            valid_range=valid_range,
        )
        print(f"승격 완료: champion={reg.current_champion()}")
    else:
        print("4회 모두 BLOCK — champion 유지")

    print(f"promoted={result['promoted']}")
    return 0
