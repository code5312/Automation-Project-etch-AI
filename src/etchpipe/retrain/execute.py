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

# ADR-0005 결정4 관문 C③(기존 val 40에서도 결정3 3조건 유지): 드리프트셋이 아니라
# 원래(비-드리프트) 데이터셋의 val 분할이 필요하다. `scripts/run_model_comparison.py`와
# 동일한 경로·해시[확인, `data/processed/` 아래 폴더 하나뿐].
VAL40_RAW_DIR = REPO_ROOT / "data" / "raw"
VAL40_GRAPHS_ROOT = REPO_ROOT / "data" / "processed" / "4b8c0ce1bc98" / "graphs"
VAL40_SPLITS_PATH = REPO_ROOT / "data" / "splits.json"


class DriftState:
    """드리프트 데이터(train/val/golden_drift)를 들고 다니며, `expand_train`(2차 시도)이
    train을 늘리면 다시 ETL해 `dataset_hash`/`graphs_root`를 갱신한다.

    모든 인자의 기본값은 W6-03/04 실제 운영 경로(train 40/val 10/golden_drift 10,
    `data/drift/`)와 동일 — `DriftState()`(무인자)는 그대로 전과 같이 동작한다[확인,
    2026-11-13 리팩터링]. `raw_dir`/`processed_root`/크기/시드/접두사를 바꾸면 W6-07
    `retrain-demo`처럼 작은 별도 드리프트셋으로도 같은 로직을 재사용할 수 있다."""

    def __init__(
        self,
        raw_dir: pathlib.Path = DRIFT_RAW_DIR,
        processed_root: pathlib.Path = DRIFT_PROCESSED_ROOT,
        n_train: int = 40,
        n_val: int = 10,
        n_golden_drift: int = 10,
        seed: int = 9100,
        prefix: str = "drift",
        substrate_a_ie: float = SUBSTRATE_A_IE,
        sim_config_id: str = "live_v1",
        expand_seed: int = 9200,
        expand_prefix: str = "driftx",
    ) -> None:
        from etchpipe.sim.drift_recipes import drift_splits

        self.raw_dir = raw_dir
        self.processed_root = processed_root
        self.substrate_a_ie = substrate_a_ie
        self.sim_config_id = sim_config_id
        self.expand_seed = expand_seed
        self.expand_prefix = expand_prefix
        self.splits, self.params_by_id = drift_splits(n_train, n_val, n_golden_drift, seed, prefix)
        self._process()

    def _process(self) -> None:
        from etchpipe.etl.etl import THETA_KEYS, process_dataset

        manifest = process_dataset(self.raw_dir, self.splits, self.processed_root, THETA_KEYS)
        self.dataset_hash = manifest["dataset_hash"]
        self.graphs_root = self.processed_root / self.dataset_hash / "graphs"

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
        extra = make_recipes(n_more, seed=self.expand_seed, prefix=self.expand_prefix)
        for r in extra:
            rid = r["run_id"]
            out = self.raw_dir / rid
            if not (out / "meta.json").exists():
                run_sim(
                    r["params"], out,
                    substrate_a_ie=self.substrate_a_ie, sim_config_id=self.sim_config_id,
                )
            self.params_by_id[rid] = r["params"]
            self.splits["train"].append(rid)
        self._process()
        return n_more


def make_fine_tune_fn(state: DriftState, fallback_norm, out_root: pathlib.Path = RETRAIN_RUNS_ROOT):
    def fine_tune_fn(champion_ckpt: pathlib.Path, cfg: dict) -> pathlib.Path:
        from etchpipe.eval.eval import load_model_a
        from etchpipe.train.loop import train_one_epoch

        model, _note, ckpt_norm = load_model_a(champion_ckpt)
        norm = ckpt_norm or fallback_norm
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR * cfg["lr_mult"])
        epochs = cfg["epochs"][0]  # 하한(모듈 docstring 참고)

        out_dir = out_root / f"attempt{cfg['attempt']}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        history = []
        # W7-07: 전체 파이프라인 중 fine-tune 단계가 압도적 병목(실측 92%)이었다 — 같은
        # train_ids의 그래프 npz를 epochs번 그대로 재읽던 것을 에폭 간 공유 캐시로 생략.
        graph_cache: dict = {}
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_one_epoch(
                model, optimizer, state.graphs_root, state.splits["train"],
                norm=norm, device="cpu", cache=graph_cache,
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


def _eval_recipes(
    model,
    norm,
    run_ids: list[str],
    graphs_root: pathlib.Path,
    raw_dir: pathlib.Path,
    check_shape: bool,
) -> dict:
    """레시피 묶음의 깊이·CD(아직 안 배선, 깊이로 대체 — [미확인, 알려진 한계])·롤아웃
    스텝별 Chamfer·물리(선 꼬임)를 모은다. `check_shape=True`면 IoU·HD95도 같이(ADR-0005
    결정4 Gate C①②용 — 비용이 들어서 필요할 때만 켠다)."""
    from etchpipe.train.rollout import rollout_val_error_by_step

    depth_pred, depth_true, rollout_errors = [], [], []
    iou_vals, hd95_vals, self_x_total = [], [], 0
    for rid in run_ids:
        meta = json.loads((pathlib.Path(raw_dir) / rid / "meta.json").read_text())
        actual_k = meta["actual_k"]
        steps = sorted({s for s in EVAL_STEPS if s <= actual_k} | {actual_k})
        result = rollout_val_error_by_step(
            model, graphs_root, rid, steps, meta["step_min"], norm=norm,
            check_physics=True, check_shape=check_shape,
            fix_mask=True, guard=True, loop_removal=True,
        )
        final = result[actual_k]
        depth_pred.append(final["depth_pred"])
        depth_true.append(final["depth_true"])
        rollout_errors.append([result[s]["chamfer"] for s in steps])
        self_x_total += final["self_intersections"]
        if check_shape:
            iou_vals.append(final["iou"])
            hd95_vals.append(final["hd95"])
    return {
        "depth_pred": depth_pred, "depth_true": depth_true,
        # [미확인, 알려진 한계] CD 미배선 — 깊이 값 복제(run_gate_negative.py와 동일 관례)
        "cd_pred": depth_pred, "cd_true": depth_true,
        "rollout_errors": rollout_errors, "self_x_total": self_x_total,
        "iou": iou_vals, "hd95": hd95_vals,
    }


def _val40_run_ids() -> list[str]:
    return json.loads(VAL40_SPLITS_PATH.read_text())["val"]


def make_gate_fn(state: DriftState, champion_ckpt: pathlib.Path = CHAMPION_CKPT):
    """ADR-0005 — 관문 A(결정3, 레시피별 깊이·CD·롤아웃 3조건 통과율) + 물리(ADR-0004
    결정4, 변경 없음) + 관문 C(결정4, 윌콕슨 IoU 회귀·HD95 악화·val 40 재확인) — **관문
    B(개선 검정)는 범위 제외**(ADR 재결정, "정기 재학습" 코드패스 없음).

    챔피언은 이 게이트 함수가 재학습 루프 전체(최대 4회 시도)에서 재사용되는 동안
    바뀌지 않으므로, 챔피언의 드리프트셋 IoU·HD95는 **클로저 생성 시 한 번만** 계산해
    캐시한다(시도마다 다시 평가하지 않음 — 비용 절감)[제안, 2026-11-14]."""
    from etchpipe.eval.eval import load_model_a

    champion_model, _note, champion_norm = load_model_a(champion_ckpt)
    champion_drift_eval = _eval_recipes(
        champion_model, champion_norm, state.eval_ids(), state.graphs_root, state.raw_dir,
        check_shape=True,
    )

    def gate_fn(ckpt_path: pathlib.Path) -> dict:
        from etchpipe.eval.eval import load_model_a
        from etchpipe.gate.gate import check_no_regression, check_recipe_pass_rate

        model, _note, ckpt_norm = load_model_a(ckpt_path)
        gate_cfg = yaml.safe_load(GATE_CFG_PATH.read_text())["gate"]

        cand = _eval_recipes(
            model, ckpt_norm, state.eval_ids(), state.graphs_root, state.raw_dir, check_shape=True
        )
        gate_a_ok, gate_a_detail = check_recipe_pass_rate(
            cand["depth_pred"], cand["depth_true"], cand["cd_pred"], cand["cd_true"],
            cand["rollout_errors"],
            grid=gate_cfg["grid"], max_dim_pct=gate_cfg["max_dim_pct"],
            cd_floor=gate_cfg["cd_floor"], rollout_max_ratio=gate_cfg["rollout_max_ratio"],
            rollout_min_increase=gate_cfg["rollout_min_increase"],
            pass_rate=gate_cfg["recipe_pass_rate"],
        )
        physics_ok = cand["self_x_total"] == 0

        gate_c_ok, gate_c_detail = True, {}
        if gate_a_ok and physics_ok:
            # Gate C③(val 40 재확인)은 비용이 커서 A+물리를 통과했을 때만 돈다.
            val_eval = _eval_recipes(
                model, ckpt_norm, _val40_run_ids(), VAL40_GRAPHS_ROOT, VAL40_RAW_DIR,
                check_shape=False,
            )
            val_pass = check_recipe_pass_rate(
                val_eval["depth_pred"], val_eval["depth_true"],
                val_eval["cd_pred"], val_eval["cd_true"], val_eval["rollout_errors"],
                grid=gate_cfg["grid"], max_dim_pct=gate_cfg["max_dim_pct"],
                cd_floor=gate_cfg["cd_floor"], rollout_max_ratio=gate_cfg["rollout_max_ratio"],
                rollout_min_increase=gate_cfg["rollout_min_increase"],
                pass_rate=gate_cfg["recipe_pass_rate"],
            )
            gate_c_ok, gate_c_detail = check_no_regression(
                cand["iou"], champion_drift_eval["iou"], cand["hd95"], champion_drift_eval["hd95"],
                val_pass_rate_result=val_pass, alpha=gate_cfg["alpha_retrain"],
            )

        passed = gate_a_ok and physics_ok and gate_c_ok
        reasons = []
        if not gate_a_ok:
            rate, req = gate_a_detail["pass_rate"], gate_cfg["recipe_pass_rate"]
            reasons.append(f"recipe pass rate {rate:.2f} < {req}")
        if not physics_ok:
            reasons.append(f"physics violation: self_intersections={cand['self_x_total']}")
        if gate_a_ok and physics_ok and not gate_c_ok:
            reasons.append(f"regression check (gate C) failed: {gate_c_detail}")

        return {
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "reasons": reasons,
            "reference_metrics": {"gate_a": gate_a_detail, "gate_c": gate_c_detail},
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
