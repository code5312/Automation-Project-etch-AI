"""W6-08: `/predict` 실모델 추론 — champion 로드(레지스트리), t=0 메시 구성(ViennaPS,
순수 기하 — 실제 식각 시뮬레이션은 안 함), 자기회귀 롤아웃, cd/depth/taper 계산.
`app.py`(FastAPI 라우팅)와 분리해 viennaps/physicsnemo 의존을 한 곳에 모은다 —
`ps.MakeTrench`는 함수 안에서 지연 임포트(viennaps 없는 환경에서도 이 모듈의 나머지는
import 가능하게, `eval.load_model_a`의 physicsnemo 지연 임포트와 같은 이유).

**t=0 메시를 ViennaPS로 만드는 이유** [확인, 2026-11-12]: 실제 t=0 점 밀도·분포는
ViennaLS 레벨셋 메싱 결과라(레시피별 점 개수가 다름, 실측 확인) 순수 Python으로
재현할 수 없다. `MakeTrench`는 기하 구성만 하고 `SF6O2Etching` 프로세스는 적용하지
않아 빠르다(ray tracing 없음).
"""

import pathlib
import tempfile

import numpy as np
import torch

from etchpipe.etl.etl import (
    THETA_KEYS,
    auto_yr,
    load_snapshot,
    outer_path,
    remove_self_crossing_loops,
    to_graph,
)
from etchpipe.eval.eval import load_model_a
from etchpipe.eval.metrics import cd_profile, depth, mask_from_points, taper_angle_deg
from etchpipe.registry.registry import Registry
from etchpipe.train.rollout import lines_from_edge_index, rollout

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REGISTRY_ROOT = REPO_ROOT / "registry"

GRID_DELTA = 0.3  # configs/sim/base.yaml sim.grid_delta — 학습 데이터 생성과 동일 조건
STEP_MIN = 1.0  # run_sim.py STEP_MIN과 동일
MAX_STEPS = 20  # configs/sim/base.yaml sim.K — 학습 데이터 상한, 그 이상은 외삽이라 자름
XR = (-15.0, 15.0)

# model_id -> 체크포인트 경로. 카드는 artifact_sha256만 기록하므로(설계상 경로 자체는
# 저장 안 함) 여기서 짧게 고정 매핑한다 [제안, 지금은 champion 1개뿐이라 단순하게 둠 —
# W6-04 루프가 실제로 새 champion을 승격하기 시작하면 이 매핑도 늘어나야 함].
_ARTIFACT_PATHS = {"mgn_tuned_ep2": REPO_ROOT / "runs" / "mgn_tuned" / "model.pt"}


class ChampionUnavailable(RuntimeError):
    """레지스트리에 champion이 없거나(아직 등록 전) 무결성 검증에 실패했을 때."""


_cache: dict[str, dict] = {}


def load_champion() -> dict:
    """프로세스당 한 번만 로드하고 캐시(서빙은 추론마다 체크포인트를 다시 읽지 않는다)."""
    if "champion" in _cache:
        return _cache["champion"]
    reg = Registry(REGISTRY_ROOT)
    model_id = reg.current_champion()
    if model_id is None:
        raise ChampionUnavailable("레지스트리에 등록된 champion이 없습니다.")
    if model_id not in _ARTIFACT_PATHS:
        raise ChampionUnavailable(f"model_id {model_id!r}의 체크포인트 경로를 모릅니다.")
    artifact_path = _ARTIFACT_PATHS[model_id]
    if not reg.verify_artifact(model_id, artifact_path):
        raise ChampionUnavailable(f"아티팩트 무결성 검증 실패: {model_id}")
    card = reg.load_card(model_id)
    model, _note, norm = load_model_a(artifact_path)
    _cache["champion"] = {"model_id": model_id, "model": model, "norm": norm, "card": card}
    return _cache["champion"]


def build_initial_graph(
    mask_width: float, mask_height: float, theta: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`ps.MakeTrench`로 t=0 표면을 만들고 `etl.to_graph`로 그래프화."""
    import viennaps as ps  # 지연 임포트(모듈 docstring 참고)

    ps.Length.setUnit("um")
    ps.Time.setUnit("min")
    domain = ps.Domain(xExtent=30, gridDelta=GRID_DELTA)
    ps.MakeTrench(
        domain, trenchWidth=mask_width, trenchDepth=0.0,
        maskHeight=mask_height, maskTaperAngle=10,
    ).apply()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "t000.vtp"
        domain.saveSurfaceMesh(str(path), addInterfaces=True)
        p, lines, mid, nv = load_snapshot(path)
    node, ei, edge = to_graph(p, lines, mid, nv, theta, t=0.0)
    return (
        torch.from_numpy(node), torch.from_numpy(ei), torch.from_numpy(edge), torch.from_numpy(p),
    )


def is_in_range(theta_dict: dict[str, float], valid_range: dict[str, dict[str, float]]) -> bool:
    return all(valid_range[k]["min"] <= v <= valid_range[k]["max"] for k, v in theta_dict.items())


def predict_recipe(theta_dict: dict[str, float], times: list[float]) -> dict:
    """체크리스트 W6-08: 레시피 하나 + 요청 시간들 → 프로파일(시간별 표면선) + cd/depth/
    taper_deg(마지막 요청 시간 기준) + in_range(경고만, 거부 안 함 — DOC-01).

    **시간 해석** [제안, 설계서 §7 예시는 0.25/0.5 같은 분(min) 단위 연속값을 쓰지만,
    모델은 `step_min`(1분) 단위로만 학습됐다]: 각 요청 시간을 가장 가까운 정수
    스텝(`round(t/STEP_MIN)`, 최소 1)으로 반올림하고, `MAX_STEPS`(학습 데이터 상한)를
    넘으면 잘라낸다(`clamped=True`로 표시) — 분 단위 보간은 검증된 적 없어 하지 않는다."""
    champion = load_champion()
    model, norm, card = champion["model"], champion["norm"], champion["card"]

    theta = np.array([theta_dict[k] for k in THETA_KEYS], dtype=np.float32)
    node0, edge_index, edge0, pos0 = build_initial_graph(
        theta_dict["mask_width"], theta_dict["mask_height"], theta
    )
    lines = lines_from_edge_index(edge_index)

    raw_steps = [max(1, round(t / STEP_MIN)) for t in times]
    clamped = any(s > MAX_STEPS for s in raw_steps)
    step_for_time = [min(s, MAX_STEPS) for s in raw_steps]
    n_steps = max(step_for_time) if step_for_time else 0

    positions = rollout(
        model, node0, edge_index, edge0, pos0, n_steps, STEP_MIN, norm,
        fix_mask=True, guard=True,
    )

    profiles = []
    last_pos = pos0.numpy()
    for s in step_for_time:
        pos_s = positions[s].numpy()
        curve = remove_self_crossing_loops(outer_path(pos_s, lines))
        profiles.append({"t": round(s * STEP_MIN, 6), "points": curve.tolist()})
        last_pos = pos_s

    last_curve = remove_self_crossing_loops(outer_path(last_pos, lines))
    depth_v = depth(last_curve)
    mask = mask_from_points(last_pos, lines, xr=XR)
    cd_prof = cd_profile(mask, XR, auto_yr(), depth_v)
    taper = taper_angle_deg(cd_prof["top"], cd_prof["bot"], depth_v)

    return {
        "model_version": champion["model_id"],
        "profiles": profiles,
        "cd": cd_prof["mid"],
        "depth": depth_v,
        "taper_deg": taper,
        "in_range": is_in_range(theta_dict, card["valid_range"]),
        "clamped": clamped,
    }
