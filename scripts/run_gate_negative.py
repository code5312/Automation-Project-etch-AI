"""W5-06/W5-08: gate-negative 부정 테스트 6종 — 실모델 기준(ADR-0004 이후, 2026-11-12).

W2-10 더미 모델(`RandomDisplacementModel` 등) 프로토타입을 실모델로 교체한다. 설계서 §3
C5 자체 테스트 6종: 정상 통과 / 노이즈 모델 차단 / 1에폭(저학습) 모델 차단 / 시뮬레이터
버전 변경 데이터로 학습한 모델 차단 / `dataset_hash` 불일치(코드 2) / 골든 변조(코드 2).

**데이터는 `golden/raw/`(git에 커밋됨, CI에서도 접근 가능)를 쓴다** [확인, 2026-11-12]:
`data/raw`·`data/processed`는 `/data/` 전체가 `.gitignore`에 걸려 CI 체크아웃에는 없다
(CLAUDE.md "데이터 커밋 금지"). golden 10개 레시피는 ADR-0004 채점 기준 자체가 가리키는
표준 평가셋이기도 해서 의미상으로도 더 맞다 — `etl_run`으로 그 자리에서 그래프로
변환한다(`tests/unit/test_rollout.py`가 fixture에 쓰는 것과 같은 방식).

**체크포인트는 `runs/mgn_tuned/model.pt` 1개만 git에 커밋했다**(`.gitignore` 예외,
2026-11-12 사용자 결정) — `/runs/`·`*.pt` 블랭킷 규칙에 대한 유일한 예외. "1에폭(저학습)
모델" 케이스에 쓰는 실제 체크포인트(`runs/20261110T020341143621_c07d1b/ckpt_last.pt`)는
커밋 안 했다 — 로컬에만 있을 때는 그걸 그대로 쓰고, 없으면(CI) `NoisyDisplacementModel`
합성 대체로 넘어간다(아래).

**'시뮬레이터 버전 변경 데이터로 학습한 모델'은 실제 체크포인트가 없다** [미확인 → 합성
대체, 2026-11-11 사용자 결정]: ViennaLS 5.9.0은 import 자체가 안 돼(CLAUDE.md 환경 제약)
그 버전으로 실제 학습하는 게 불가능하다. 대신 `SyntheticDepthBiasModel`로 mgn_tuned의
실제 예측 변위 중 깊이(y) 방향 성분을 상수 배율로 부풀려 "그리드/버전이 바뀐 데이터로
학습했다면 생길 법한" 깊이 체계적 오차를 모사한다 — **실제로 그 데이터로 학습한 모델이
아니라는 점을 결과에 명시**한다.

**CD(중간 선폭) 신호는 아직 파이프라인에 안 배선돼 있다** [미확인, 알려진 한계]:
`check_cd_depth_error`는 깊이·CD를 각각 받아야 하는데, `rollout_val_error_by_step`은
아직 CD를 계산하지 않는다(`metrics.cd_at_y`/`cd_profile`은 W5-01/02에서 추가됐지만 아직
어떤 실행 경로도 호출하지 않음). 이 스크립트는 `cd_pred`/`cd_true`에 `depth_pred`/
`depth_true`를 그대로 복제해서 넘긴다 — 즉 ②의 CD 쪽은 깊이 판정과 같은 신호를 중복
보는 것일 뿐 독립적인 CD 검증이 아니다. CD 배선은 별도 항목으로 남긴다.

**ADR-0005 2단계 재검증(2026-11-14 추가)**: 실제 드리프트 승격 경로
(`retrain/execute.py::make_gate_fn`)는 더 이상 `run_gate`를 안 쓰고
`check_recipe_pass_rate`(관문 A)·`check_no_regression`(관문 C, 관문 B 제외)를 직접
쓴다 — 아래 6종 케이스를 `run_gate`(기존)와 **새 함수 양쪽으로 똑같이 재실행**해서
새 경로도 같은 기준(정상 PASS, 나머지 BLOCK)을 지키는지 검증한다. 관문 C③(val 40
재확인)은 CI에 없는 `data/raw`가 필요해 이 스크립트에서는 건너뛴다(`check_no_regression`
의 `val_pass_rate_result=None` 설계대로 — 건너뛰면 그 조건은 자동 통과로 본다는 뜻).
해시 위조 2종(케이스 5·6)은 `run_gate` 자체의 기능이라 새 경로에 대응 항목이 없다
(그대로 유지, 재검증 대상 아님).

**실측 결과 중 하나는 "정상 모델"의 기대값 자체가 경로별로 다르다**[확인, 2026-11-14,
사용자 결정]: `mgn_tuned`는 golden 10개(중심5·경계3·극단2, 일부러 어려운 표본)에서
레시피별 3조건을 8/10(80%)만 통과해 ADR-0005 신규 경로에서는 BLOCK이 기대값이다 —
기존 `run_gate`(평균 집계)는 이 2개의 실패를 평균에 묻어 PASS를 낸다. 버그가 아니라
"평균 집계가 개별 레시피 실패를 가린다"는 ADR-0005의 채택 근거가 그대로 실측 확인된
것. 실제 승격 판정에 쓰는 세트(golden_drift10+val10=20개, val 40개)에서는 이 문제가
재현되지 않을 가능성이 높다 — 90% 기준 자체가 val 40 기준 Clopper-Pearson으로 도출된
값이라 N=10·편향 표본에는 원래 안 맞는다."""

import json
import pathlib
import sys
import tempfile

import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from etchpipe.etl.etl import THETA_KEYS, etl_run  # noqa: E402
from etchpipe.eval.eval import load_model_a  # noqa: E402
from etchpipe.gate.gate import (  # noqa: E402
    GateResult,
    check_no_regression,
    check_recipe_pass_rate,
    run_gate,
)
from etchpipe.models.meshgraphnet import MeshGraphNetModel  # noqa: E402
from etchpipe.train.rollout import rollout_val_error_by_step  # noqa: E402

DATASET_HASH = "4b8c0ce1bc98"  # configs/model/mgn_v1.yaml, reports/evidence/W3-04
GOLDEN_HASH = "e95bdca4de55"  # golden/golden.json, scripts/make_golden.py
GOLDEN_RAW_DIR = REPO_ROOT / "golden" / "raw"
GATE_CFG_PATH = REPO_ROOT / "configs" / "pipeline" / "gate.yaml"
MGN_TUNED_CKPT = REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"
EARLY_CKPT = REPO_ROOT / "runs" / "20261110T020341143621_c07d1b" / "ckpt_last.pt"

STEPS = (5, 10, 15, 20)


class NoisyDisplacementModel(torch.nn.Module):
    """CI 전용 대체재 — 1에폭(저학습) 체크포인트가 저장소에 없을 때 쓴다(로컬 실측은
    `runs/20261110T020341143621_c07d1b/ckpt_last.pt`, [확인] 아래 `build_cases`가
    존재 여부로 분기). mgn_tuned 변위에 시드 고정 가우시안 노이즈를 더해 "학습이 덜
    끝난 모델"의 증상(일관된 편향이 아니라 들쭉날쭉한 변위)을 모사한다[제안, 합성 대체]."""

    def __init__(self, base_model: torch.nn.Module, std: float, seed: int = 0) -> None:
        super().__init__()
        self.base_model = base_model
        self.std = std
        self._gen = torch.Generator().manual_seed(seed)

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        disp = self.base_model(node, edge_index, edge)
        noise = torch.randn(disp.shape, generator=self._gen) * self.std
        return disp + noise


class SyntheticDepthBiasModel(torch.nn.Module):
    """실제 체크포인트 아님 — mgn_tuned 변위의 깊이(y) 성분을 `y_scale`배로 부풀려
    시뮬레이터 버전/그리드 변경 데이터로 학습했을 때 생길 법한 깊이 체계적 오차를 모사
    [제안, 합성 대체, 2026-11-11 사용자 결정]."""

    def __init__(self, base_model: torch.nn.Module, y_scale: float) -> None:
        super().__init__()
        self.base_model = base_model
        self.y_scale = y_scale

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        disp = self.base_model(node, edge_index, edge)
        scaled = disp.clone()
        scaled[:, 1] = scaled[:, 1] * self.y_scale
        return scaled


def _golden_recipe_ids() -> list[str]:
    return sorted(p.name for p in GOLDEN_RAW_DIR.iterdir() if p.is_dir())


def _gate_thresholds() -> dict:
    return yaml.safe_load(GATE_CFG_PATH.read_text())["gate"]


def evaluate_model(
    model, norm, graphs_root: pathlib.Path,
    fix_mask: bool = True, guard: bool = True, loop_removal: bool = True,
) -> dict:
    """golden 10개 레시피(이미 `graphs_root`에 변환돼 있음)에 대해 `run_gate`/ADR-0005
    새 함수(`check_recipe_pass_rate`/`check_no_regression`) 양쪽에 다 넘길 수 있는
    입력을 구성 — `check_shape=True`로 IoU·HD95도 같이 낸다(ADR-0005 2단계 재검증용,
    2026-11-14 추가)."""
    depth_pred, depth_true = [], []
    chamfer_by_step_per_recipe = []
    iou_vals, hd95_vals = [], []
    self_x_total = 0
    for rid in _golden_recipe_ids():
        meta = json.loads((GOLDEN_RAW_DIR / rid / "meta.json").read_text())
        actual_k = meta["actual_k"]
        run_steps = sorted({s for s in STEPS if s <= actual_k} | {actual_k})
        result = rollout_val_error_by_step(
            model, graphs_root, rid, run_steps, meta["step_min"], norm=norm,
            check_physics=True, check_shape=True,
            fix_mask=fix_mask, guard=guard, loop_removal=loop_removal,
        )
        chamfer_by_step_per_recipe.append([result[s]["chamfer"] for s in run_steps])
        final = result[actual_k]
        depth_pred.append(final["depth_pred"])
        depth_true.append(final["depth_true"])
        self_x_total += final["self_intersections"]
        iou_vals.append(final["iou"])
        hd95_vals.append(final["hd95"])
    return {
        "depth_pred": depth_pred,
        "depth_true": depth_true,
        "cd_pred": depth_pred,  # [미확인, 알려진 한계] CD 미배선 — 깊이 값 복제(독립 신호 아님)
        "cd_true": depth_true,
        "rollout_step_errors_by_recipe": chamfer_by_step_per_recipe,
        "self_intersection_count": self_x_total,
        "iou": iou_vals,
        "hd95": hd95_vals,
    }


def _for_run_gate(inputs: dict) -> dict:
    """`run_gate`는 `iou`/`hd95` 인자를 모른다 — `evaluate_model`이 양쪽 경로용으로
    같이 반환하는 값 중 기존 경로에 안 맞는 키만 뺀다."""
    return {k: v for k, v in inputs.items() if k not in ("iou", "hd95")}


def _new_path_gate(gate_cfg: dict, champion_inputs: dict, candidate_inputs: dict) -> GateResult:
    """ADR-0005 2단계 재검증용 — 실제 `retrain/execute.py::make_gate_fn`과 같은 순서로
    관문 A(결정3) → 물리 → 관문 C(결정4, 관문 B 제외)를 적용한다. 관문 C③(val 40)은
    이 스크립트엔 `data/raw`가 없어(CI 제약) 건너뛴다(`val_pass_rate_result=None`)."""
    gate_a_ok, gate_a_detail = check_recipe_pass_rate(
        candidate_inputs["depth_pred"], candidate_inputs["depth_true"],
        candidate_inputs["cd_pred"], candidate_inputs["cd_true"],
        candidate_inputs["rollout_step_errors_by_recipe"],
        grid=gate_cfg["grid"], max_dim_pct=gate_cfg["max_dim_pct"], cd_floor=gate_cfg["cd_floor"],
        rollout_max_ratio=gate_cfg["rollout_max_ratio"],
        rollout_min_increase=gate_cfg["rollout_min_increase"],
        pass_rate=gate_cfg["recipe_pass_rate"],
    )
    physics_ok = candidate_inputs["self_intersection_count"] == 0

    gate_c_ok, gate_c_detail = True, {}
    if gate_a_ok and physics_ok:
        gate_c_ok, gate_c_detail = check_no_regression(
            candidate_inputs["iou"], champion_inputs["iou"],
            candidate_inputs["hd95"], champion_inputs["hd95"],
            val_pass_rate_result=None, alpha=gate_cfg["alpha_retrain"],
        )

    passed = gate_a_ok and physics_ok and gate_c_ok
    reasons = []
    if not gate_a_ok:
        rate, req = gate_a_detail["pass_rate"], gate_cfg["recipe_pass_rate"]
        reasons.append(f"recipe pass rate {rate:.2f} < {req}")
    if not physics_ok:
        n_x = candidate_inputs["self_intersection_count"]
        reasons.append(f"physics violation: self_intersections={n_x}")
    if gate_a_ok and physics_ok and not gate_c_ok:
        reasons.append(f"regression check (gate C) failed: {gate_c_detail}")

    return GateResult(
        passed=passed, exit_code=0 if passed else 1, reasons=reasons,
        reference_metrics={"gate_a": gate_a_detail, "gate_c": gate_c_detail},
    )


def build_cases(graphs_root: pathlib.Path) -> list[dict]:
    gate_cfg = _gate_thresholds()
    gate_kwargs = {
        "max_chamfer": gate_cfg["max_chamfer"],
        "grid": gate_cfg["grid"],
        "max_dim_pct": gate_cfg["max_dim_pct"],
        "rollout_max_ratio": gate_cfg["rollout_max_ratio"],
    }
    golden_pairs: list[tuple] = []  # ①은 참고 지표만(ADR-0004) — 실모델 테스트에선 비움

    mgn_tuned, _, mgn_tuned_norm = load_model_a(MGN_TUNED_CKPT)

    cases = []
    champion_inputs = evaluate_model(mgn_tuned, mgn_tuned_norm, graphs_root)

    def _add_pair(
        name: str,
        expected_passed: bool,
        old_result: GateResult,
        inputs: dict,
        expected_passed_new: bool | None = None,
    ) -> None:
        cases.append({
            "name": f"{name} [기존 run_gate]",
            "expected_passed": expected_passed,
            "result": old_result,
        })
        new_result = _new_path_gate(gate_cfg, champion_inputs, inputs)
        new_expected = expected_passed if expected_passed_new is None else expected_passed_new
        cases.append({
            "name": f"{name} [ADR-0005 신규 경로]",
            "expected_passed": new_expected,
            "result": new_result,
        })

    # [확인, 2026-11-14 실측] 정상 모델(mgn_tuned)은 golden 10개 레시피(중심5·경계3·
    # 극단2로 일부러 어려운 케이스를 섞은 작은 표본)에서 10개 중 8개(80%)만 레시피별
    # 3조건을 통과한다 — 기존 run_gate(평균 집계)는 이 2개의 실패를 평균에 묻어 PASS를
    # 내지만, ADR-0005 레시피별 판정은 그대로 드러내 90% 기준에서 BLOCK된다. 버그가
    # 아니라 설계대로 동작한 것[사용자 결정 2026-11-14]: golden-10은 실제 승격 판정에
    # 쓰는 세트(golden_drift10+val10=20개, val 40개)가 아니라 CI에 커밋된 유일한
    # 데이터라 이 재검증에 쓸 뿐이고, 90% 기준 자체는 val 40 기준 Clopper-Pearson으로
    # 도출된 값이라 N=10·편향 표본에 그대로 적용하면 이렇게 불안정하게 나올 수 있다.
    # "신규 경로"만 BLOCK(80%)을 기대값으로 삼는다.
    _add_pair(
        "정상 모델(mgn_tuned)", True,
        run_gate(mgn_tuned, golden_pairs, **gate_kwargs, **_for_run_gate(champion_inputs)),
        champion_inputs,
        expected_passed_new=False,
    )

    noise_model = MeshGraphNetModel(processor_size=6, hidden_dim_processor=128)
    noise_model.eval()
    inputs = evaluate_model(noise_model, mgn_tuned_norm, graphs_root)
    _add_pair(
        "노이즈 모델(무작위 가중치, 미학습)", False,
        run_gate(noise_model, golden_pairs, **gate_kwargs, **_for_run_gate(inputs)), inputs,
    )

    if EARLY_CKPT.exists():
        early_model, early_note, early_norm = load_model_a(EARLY_CKPT)
        early_case_norm = early_norm or mgn_tuned_norm
        early_name = f"저학습 체크포인트({early_note}, 로컬 실측)"
    else:
        early_model = NoisyDisplacementModel(mgn_tuned, std=2.0)
        early_case_norm = mgn_tuned_norm
        early_name = "저학습 모델(합성 대체 — CI에 체크포인트 없음, 로컬 실측은 W5-09 증거 참고)"
    inputs = evaluate_model(early_model, early_case_norm, graphs_root)
    _add_pair(
        f"{early_name}, 설계서 '1에폭' 모델 대체재", False,
        run_gate(early_model, golden_pairs, **gate_kwargs, **_for_run_gate(inputs)), inputs,
    )

    sim_version_model = SyntheticDepthBiasModel(mgn_tuned, y_scale=1.6)
    inputs = evaluate_model(sim_version_model, mgn_tuned_norm, graphs_root)
    _add_pair(
        "시뮬레이터 버전 변경 데이터로 학습한 모델(합성 대체 — 실제 체크포인트 아님)", False,
        run_gate(sim_version_model, golden_pairs, **gate_kwargs, **_for_run_gate(inputs)), inputs,
    )

    r5 = run_gate(
        mgn_tuned, golden_pairs, max_chamfer=gate_cfg["max_chamfer"],
        expected_dataset_hash=DATASET_HASH, actual_dataset_hash="tampered",
    )
    cases.append({"name": "dataset_hash 불일치", "expected_exit_code": 2, "result": r5})

    r6 = run_gate(
        mgn_tuned, golden_pairs, max_chamfer=gate_cfg["max_chamfer"],
        expected_golden_hash=GOLDEN_HASH, actual_golden_hash="tampered",
    )
    cases.append({"name": "골든셋 변조", "expected_exit_code": 2, "result": r6})

    return cases


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        graphs_root = pathlib.Path(tmp) / "graphs"
        for rid in _golden_recipe_ids():
            etl_run(GOLDEN_RAW_DIR / rid, graphs_root / rid, THETA_KEYS)
        cases = build_cases(graphs_root)

    all_ok = True
    report = []
    for c in cases:
        result = c["result"]
        if "expected_exit_code" in c:
            ok = result.exit_code == c["expected_exit_code"]
            exp_desc = f"exit_code={c['expected_exit_code']}"
        else:
            ok = result.passed == c["expected_passed"]
            exp_desc = f"passed={c['expected_passed']}"
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] {c['name']}: 기대={exp_desc} 실제=passed:{result.passed},"
            f" exit_code:{result.exit_code} reasons={result.reasons}"
        )
        report.append({
            "name": c["name"],
            "expected": exp_desc,
            "actual_passed": result.passed,
            "actual_exit_code": result.exit_code,
            "passed_as_expected": ok,
            "reasons": result.reasons,
            "reference_metrics": result.reference_metrics,
        })

    out = pathlib.Path("build/gate_negative_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    if not all_ok:
        print("\ngate-negative 실패: 기대와 다른 case가 있음")
        return 1
    print(f"\ngate-negative 통과: {len(cases)}개 케이스 모두 기대대로 동작")
    return 0


if __name__ == "__main__":
    sys.exit(main())
