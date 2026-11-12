"""C3: 평가용 자기회귀 롤아웃(W4-05) — 팀포싱(W4-02, `loop.py`)과 다른 추론 방식.

입력(t_0) 그래프에서 시작해 모델 자신의 예측을 다음 스텝 입력으로 다시 먹인다
(자기회귀). 매 스텝:
  - **연결(edge_index) 유지**: 점 개수·순서·연결은 안 바뀐다고 가정(식각 중 재메싱 없음).
    `mgn_tuned/inference.py`의 "점 채우기"(선분이 길어지면 중간에 점을 추가하는 재메싱)는
    이 가정과 정면으로 충돌해서 **포팅하지 않았다** — ADR-0004 후속 배선(2026-11-11)에서
    사용자에게 별도 확인 필요로 남겨둠 [미확인, 알려진 한계].
  - **법선·엣지 재계산**: 점이 움직였으니 둘 다 새 `pos`에서 다시 계산해야 한다. 엣지는
    `etl.to_graph`와 같은 정의(rel_x, rel_y, norm). 법선은 세그먼트 방향의 수직 벡터를
    꼭짓점별로 평균하는데, 수직 방향은 두 가지(±90도)라 **이전 스텝 법선과 같은 쪽**을
    고르는 식으로 부호를 고정한다(회전이 매 스텝 작아서 방향이 뒤집히지 않는다고 가정
    — ETL이 VTK `Normals`에서 구하는 "진짜" 바깥쪽 방향과 달리, 여기서는 그 방향을
    다시 풀 수 없어 연속성으로 대신함).
  - **재료(material one-hot) 유지**: 점이 중간에 재료를 바꾸지 않는다고 보고 입력 그래프
    값을 그대로 들고 간다(재계산 안 함).
  - **마스크 고정**(`fix_mask`, 기본 꺼짐) / **교차가드**(`guard`, 기본 꺼짐) [확인,
    2026-11-11 — `mgn_tuned/inference.py` `rollout()`의 (3)(4) 그대로 포팅]. 둘 다 기본은
    꺼져 있다(기존 호출부·테스트 호환) — 체크포인트가 이 추론 방식으로 튜닝됐을 때만
    켠다(`eval.main`에서 명시적으로 전달).
  - **t 증가**: `etl.etl_run`과 같은 정의(`k * step_min`)로 스텝마다 `step_min`씩 더한다.

node 채널 레이아웃은 `etl.to_graph` 그대로: pos[0:2], normal[2:4], material[4:6],
theta[6:13], t[13:14] (2+2+2+7+1=14).
"""

from pathlib import Path

import numpy as np
import torch

from etchpipe.etl.etl import outer_path, remove_self_crossing_loops
from etchpipe.eval.metrics import chamfer_distance, depth, depth_error, self_intersection_count
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


def _segments_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(E,E) bool — 선분 i(a[i]->b[i])와 선분 j가 서로 가로지르면 True [확인, mgn_tuned
    `inference.py`의 `_cross_matrix` 그대로 포팅]."""

    def cr(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])

    pa, pb, pc, pd = a.unsqueeze(1), b.unsqueeze(1), a.unsqueeze(0), b.unsqueeze(0)
    return (cr(pa, pb, pc) * cr(pa, pb, pd) < 0) & (cr(pc, pd, pa) * cr(pc, pd, pb) < 0)


def _cross_pairs(pos: torch.Tensor, lines: torch.Tensor) -> set[tuple[int, int]]:
    """`lines`(E,2, 점 인덱스) 중 서로 교차하는 선분 쌍(세그먼트 인덱스, i<j) 집합."""
    x = _segments_cross(pos[lines[:, 0]], pos[lines[:, 1]])
    idx = torch.triu(x, diagonal=1).nonzero(as_tuple=False)
    return {(int(i), int(j)) for i, j in idx.tolist()}


def _apply_collision_guard(
    pos: torch.Tensor,
    disp: torch.Tensor,
    lines: torch.Tensor,
    shrink: tuple[float, ...],
) -> torch.Tensor:
    """교차가드 [확인, mgn_tuned `inference.py` `rollout()`의 (4) 그대로 포팅]. 이동으로
    새로 생긴 교차(이전에는 없었던 쌍만)에 관여한 선분의 두 끝점 변위를 `shrink` 배수로
    줄여가며 재시도 — 교차가 안 없어지면 다음 배수로 더 줄인다."""
    before = _cross_pairs(pos, lines)
    bad: set[int] = set()
    d = disp
    new_pos = pos + d
    for s in shrink:
        now = _cross_pairs(new_pos, lines) - before
        if not now:
            break
        for seg_i, seg_j in now:
            bad.add(int(lines[seg_i, 0]))
            bad.add(int(lines[seg_i, 1]))
            bad.add(int(lines[seg_j, 0]))
            bad.add(int(lines[seg_j, 1]))
        idx = sorted(bad)
        d = disp.clone()
        d[idx] = disp[idx] * s
        new_pos = pos + d
    return new_pos


def rollout(
    model,
    node0: torch.Tensor,
    edge_index: torch.Tensor,
    edge0: torch.Tensor,
    pos0: torch.Tensor,
    n_steps: int,
    step_min: float,
    norm: NodeEdgeNorm | None = None,
    fix_mask: bool = False,
    guard: bool = False,
    guard_shrink: tuple[float, ...] = (0.5, 0.25, 0.0),
) -> list[torch.Tensor]:
    """초기 그래프에서 `n_steps`만큼 자기회귀 예측. 반환: `pos` 리스트
    (길이 `n_steps+1`, `[0]`=입력 그대로, 그래디언트 없음 — 평가 전용).

    `fix_mask=True`면 마스크 노드(material[:,0]>0.5)의 변위를 매 스텝 0으로 고정하고,
    `guard=True`면 `_apply_collision_guard`로 새로 생기는 자기교차를 완화한다 — 둘 다
    기본은 꺼져 있다(기존 호출부 호환). `guard=True`면 `lines_from_edge_index`로 선분을
    한 번만 역산해 재사용(스텝마다 다시 계산 안 함, `edge_index`가 "연결 유지"로 안 바뀌어
    가능)."""
    normal = node0[:, 2:4]
    material = node0[:, 4:6]
    theta = node0[:, 6:13]
    t = node0[:, 13:14]
    pos = pos0
    edge = edge0
    positions = [pos0]
    lines = (
        torch.from_numpy(lines_from_edge_index(edge_index)).to(edge_index.device)
        if guard
        else None
    )
    with torch.no_grad():
        for _ in range(n_steps):
            node = torch.cat([pos, normal, material, theta, t], dim=1)
            node_in = normalize_node(node, norm) if norm is not None else node
            edge_in = normalize_edge(edge, norm) if norm is not None else edge
            disp = model(node_in, edge_index, edge_in)
            if fix_mask:
                is_mask = material[:, 0] > 0.5
                disp = disp.clone()
                disp[is_mask] = 0.0
            if guard:
                pos = _apply_collision_guard(pos, disp, lines, guard_shrink)
            else:
                pos = pos + disp
            normal = recompute_normals(pos, edge_index, normal)
            edge = recompute_edge_features(pos, edge_index)
            t = t + step_min
            positions.append(pos)
    return positions


def lines_from_edge_index(edge_index: torch.Tensor) -> np.ndarray:
    """`etl.to_graph`가 만든 양방향 엣지(앞쪽 절반=원래 선분, 뒤쪽 절반=반대 방향)에서
    원래 선분 `(E,2)`만 복원 [확인, `to_graph`의 `ei = concat([lines, lines[:,::-1]]).T`
    구성 그대로 역산]. `etl.outer_path`에 넘겨 표면선만 추출하는 용도(ADR-0004 물리
    규칙 배선)."""
    n_directed = edge_index.shape[1] // 2
    return edge_index[:, :n_directed].T.numpy()


def rollout_val_error_by_step(
    model,
    graphs_root,
    run_id: str,
    steps: list[int],
    step_min: float,
    norm: NodeEdgeNorm | None = None,
    check_physics: bool = False,
    fix_mask: bool = False,
    guard: bool = False,
    guard_shrink: tuple[float, ...] = (0.5, 0.25, 0.0),
    loop_removal: bool = False,
) -> dict[int, dict[str, float]]:
    """`g000.npz`(t=0)에서 `max(steps)`까지 롤아웃하고, `steps`의 각 스텝에서 실제
    `g<step>.npz`와 비교(W5-01, 설계서 C4 "스텝 5·10·20 보고") — 최종 스텝만 보면
    중간에 오차가 급증했다가 다시 줄어드는 패턴을 놓칠 수 있다
    (`reports/model_issues_mgn.md` "판단 기준" 3번).

    **중요**: Chamfer/깊이오차는 원본 메시 점 배열이 아니라 `outer_path`로 추출한
    표면선으로 계산한다 [확인, ADR-0004 후속 수정 2026-11-11 — `load_snapshot`의 점
    배열은 마스크/Si 내부 경계가 섞여 순서가 안 맞는 부분이 있어서(실측: r0000 254개
    선분 중 174개만 연속 인덱스), ADR-0004의 점-선분 Chamfer가 그 부분에서 엉뚱한
    선분을 만든다 — 원본 점 배열을 그대로 넘기면 틀린 값이 나왔던 걸 이번에 발견·수정]
    `true` 쪽은 그 스텝의 `g<step>.npz` 자체 `edge_index`로, `pred` 쪽은 `g000`의
    `edge_index`(연결 유지)로 각각 역산한다.

    `fix_mask`/`guard`/`guard_shrink`는 `rollout()`에 그대로 전달(마스크 고정·교차가드,
    둘 다 기본 꺼짐). `loop_removal=True`면 추출한 `pred` 표면선에
    `remove_self_crossing_loops`를 적용해 채점 직전에 자기교차를 지운다 [확인, mgn_tuned
    `inference.py`와 동일 — 내부 상태(`pos`)에는 반영 안 하고 채점용 출력에만 적용].

    각 스텝 결과에 `depth_pred`/`depth_true`(원값, `metrics.depth`)도 같이 낸다 [확인,
    W5-06 배선 2026-11-12 — 게이트 `check_dimension_error`가 절대오차(depth_error)가
    아니라 정답 크기 대비 %오차를 보려면 원값 두 개가 필요해서 추가].

    `check_physics=True`면 각 스텝마다 `self_intersections`(선 꼬임 수, `loop_removal`
    적용 후 기준)도 같이 낸다 [확인, ADR-0004 결정4 배선]. 예측이 도메인 밖으로 완전히
    발산해(`outer_path`가 왼쪽/오른쪽 열린 끝점을 못 찾음) 표면 자체를 복원 못 하면
    예외를 전파하지 않고 원본 점 배열로 대체 계산하고 `self_intersections=1`(명백한
    물리 위반)로 집계한다 — 토이 모델에 정규화 없이 실제 θ 크기를 그대로 먹여 1스텝에
    발산하는 경우로 실측 확인."""
    root = Path(graphs_root) / run_id
    g0 = load_graph(root / "g000.npz")
    max_step = max(steps)
    pred_lines = lines_from_edge_index(g0["edge_index"])

    positions = rollout(
        model, g0["node"], g0["edge_index"], g0["edge"], g0["pos"], max_step, step_min, norm,
        fix_mask=fix_mask, guard=guard, guard_shrink=guard_shrink,
    )
    result: dict[int, dict[str, float]] = {}
    for s in steps:
        true_graph = load_graph(root / f"g{s:03d}.npz")
        true_lines = lines_from_edge_index(true_graph["edge_index"])
        true_curve = outer_path(true_graph["pos"].numpy(), true_lines)

        pred_s = positions[s].numpy()
        diverged = False
        try:
            pred_curve = outer_path(pred_s, pred_lines)
            if loop_removal:
                pred_curve = remove_self_crossing_loops(pred_curve)
        except ValueError:
            # outer_path가 왼쪽(x<0)/오른쪽(x>0) 열린 끝점을 못 찾고 실패 — 예측이
            # 도메인 밖으로 완전히 발산해 표면 형상 자체가 깨졌다는 뜻이다. 평가
            # 배치를 멈추는 대신 원본 점 배열로 대체(부정확하지만 어차피 매우 큰
            # 오차로 나와 방향은 맞음).
            pred_curve = pred_s
            diverged = True

        entry = {
            "chamfer": chamfer_distance(pred_curve, true_curve),
            "depth_error": depth_error(pred_curve, true_curve),
            "depth_pred": depth(pred_curve),
            "depth_true": depth(true_curve),
        }
        if check_physics:
            entry["self_intersections"] = (
                1 if diverged else self_intersection_count(pred_curve)
            )
        result[s] = entry
    return result


def rollout_val_error(
    model,
    graphs_root,
    run_id: str,
    n_steps: int,
    step_min: float,
    norm: NodeEdgeNorm | None = None,
    check_physics: bool = False,
    fix_mask: bool = False,
    guard: bool = False,
    guard_shrink: tuple[float, ...] = (0.5, 0.25, 0.0),
    loop_removal: bool = False,
) -> dict[str, float]:
    """`g000.npz`(t=0)에서 `n_steps` 롤아웃한 마지막 예측을 실제 `g<n_steps>.npz`와 비교."""
    return rollout_val_error_by_step(
        model, graphs_root, run_id, [n_steps], step_min, norm, check_physics,
        fix_mask, guard, guard_shrink, loop_removal,
    )[n_steps]
