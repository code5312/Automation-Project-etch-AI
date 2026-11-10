"""C3: 평가용 자기회귀 롤아웃(W4-05) — 팀포싱(W4-02, `loop.py`)과 다른 추론 방식.

입력(t_0) 그래프에서 시작해 모델 자신의 예측을 다음 스텝 입력으로 다시 먹인다
(자기회귀). 매 스텝:
  - **연결(edge_index) 유지**: 점 개수·순서·연결은 안 바뀐다고 가정(식각 중 재메싱 없음).
  - **법선·엣지 재계산**: 점이 움직였으니 둘 다 새 `pos`에서 다시 계산해야 한다. 엣지는
    `etl.to_graph`와 같은 정의(rel_x, rel_y, norm). 법선은 세그먼트 방향의 수직 벡터를
    꼭짓점별로 평균하는데, 수직 방향은 두 가지(±90도)라 **이전 스텝 법선과 같은 쪽**을
    고르는 식으로 부호를 고정한다(회전이 매 스텝 작아서 방향이 뒤집히지 않는다고 가정
    — ETL이 VTK `Normals`에서 구하는 "진짜" 바깥쪽 방향과 달리, 여기서는 그 방향을
    다시 풀 수 없어 연속성으로 대신함).
  - **재료(material one-hot) 유지**: 점이 중간에 재료를 바꾸지 않는다고 보고 입력 그래프
    값을 그대로 들고 간다(재계산 안 함).
  - **t 증가**: `etl.etl_run`과 같은 정의(`k * step_min`)로 스텝마다 `step_min`씩 더한다.

node 채널 레이아웃은 `etl.to_graph` 그대로: pos[0:2], normal[2:4], material[4:6],
theta[6:13], t[13:14] (2+2+2+7+1=14).
"""

from pathlib import Path

import torch

from etchpipe.eval.metrics import chamfer_distance, depth_error
from etchpipe.train.loop import NodeEdgeNorm, load_graph, normalize_edge, normalize_node


def recompute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """`etl.to_graph`와 같은 정의: rel_x, rel_y, 거리."""
    rel = pos[edge_index[1]] - pos[edge_index[0]]
    dist = rel.norm(dim=1, keepdim=True)
    return torch.cat([rel, dist], dim=1)


def recompute_normals(
    pos: torch.Tensor, edge_index: torch.Tensor, prev_normal: torch.Tensor
) -> torch.Tensor:
    """세그먼트 방향의 수직 벡터를 꼭짓점별로 평균. 부호는 `prev_normal`과 같은 쪽으로 고정."""
    d = pos[edge_index[1]] - pos[edge_index[0]]
    d_unit = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
    candidate = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=1)
    ref = prev_normal[edge_index[0]]
    sign = torch.sign((candidate * ref).sum(dim=1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    candidate = candidate * sign.unsqueeze(1)

    n = pos.shape[0]
    acc = torch.zeros((n, 2), dtype=pos.dtype)
    cnt = torch.zeros(n, dtype=pos.dtype)
    src = edge_index[0]
    acc.index_add_(0, src, candidate)
    cnt.index_add_(0, src, torch.ones_like(sign))
    normal = acc / cnt.clamp_min(1).unsqueeze(1)
    return normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-9)


def rollout(
    model,
    node0: torch.Tensor,
    edge_index: torch.Tensor,
    edge0: torch.Tensor,
    pos0: torch.Tensor,
    n_steps: int,
    step_min: float,
    norm: NodeEdgeNorm | None = None,
) -> list[torch.Tensor]:
    """초기 그래프에서 `n_steps`만큼 자기회귀 예측. 반환: `pos` 리스트
    (길이 `n_steps+1`, `[0]`=입력 그대로, 그래디언트 없음 — 평가 전용)."""
    normal = node0[:, 2:4]
    material = node0[:, 4:6]
    theta = node0[:, 6:13]
    t = node0[:, 13:14]
    pos = pos0
    edge = edge0
    positions = [pos0]
    with torch.no_grad():
        for _ in range(n_steps):
            node = torch.cat([pos, normal, material, theta, t], dim=1)
            node_in = normalize_node(node, norm) if norm is not None else node
            edge_in = normalize_edge(edge, norm) if norm is not None else edge
            disp = model(node_in, edge_index, edge_in)
            pos = pos + disp
            normal = recompute_normals(pos, edge_index, normal)
            edge = recompute_edge_features(pos, edge_index)
            t = t + step_min
            positions.append(pos)
    return positions


def rollout_val_error_by_step(
    model,
    graphs_root,
    run_id: str,
    steps: list[int],
    step_min: float,
    norm: NodeEdgeNorm | None = None,
) -> dict[int, dict[str, float]]:
    """`g000.npz`(t=0)에서 `max(steps)`까지 롤아웃하고, `steps`의 각 스텝에서 실제
    `g<step>.npz`와 비교(W5-01, 설계서 C4 "스텝 5·10·20 보고") — 최종 스텝만 보면
    중간에 오차가 급증했다가 다시 줄어드는 패턴을 놓칠 수 있다
    (`reports/model_issues_mgn.md` "판단 기준" 3번)."""
    root = Path(graphs_root) / run_id
    g0 = load_graph(root / "g000.npz")
    max_step = max(steps)

    positions = rollout(
        model, g0["node"], g0["edge_index"], g0["edge"], g0["pos"], max_step, step_min, norm
    )
    result: dict[int, dict[str, float]] = {}
    for s in steps:
        true_s = load_graph(root / f"g{s:03d}.npz")["pos"].numpy()
        pred_s = positions[s].numpy()
        result[s] = {
            "chamfer": chamfer_distance(pred_s, true_s),
            "depth_error": depth_error(pred_s, true_s),
        }
    return result


def rollout_val_error(
    model,
    graphs_root,
    run_id: str,
    n_steps: int,
    step_min: float,
    norm: NodeEdgeNorm | None = None,
) -> dict[str, float]:
    """`g000.npz`(t=0)에서 `n_steps` 롤아웃한 마지막 예측을 실제 `g<n_steps>.npz`와 비교."""
    return rollout_val_error_by_step(model, graphs_root, run_id, [n_steps], step_min, norm)[
        n_steps
    ]
