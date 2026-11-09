"""C3/C4: 학습용 미분 가능 Chamfer 손실(W4-02, 설계서 §3 C3 "손실 Chamfer").

`eval/metrics.py`의 `chamfer_distance`(scipy cKDTree, 평가·게이트용)와 같은 정의(양방향
평균 최근접 거리)지만, `torch.cdist`로 계산해 역전파가 가능하다. 점 개수가 달라도 된다
(모델이 예측한 N개 점과 다음 스냅샷의 M개 점을 그대로 비교) — 스냅샷마다 점 개수가 달라서
직접 변위 MSE를 쓸 수 없는 이유(설계서 §4.1 위험)와 같다.
"""

import torch


def chamfer_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """pred: (N,2), true: (M,2). 둘 다 그래디언트가 필요 없는 쪽은 미분 안 돼도 된다."""
    d = torch.cdist(pred, true)  # (N,M)
    pred_to_true = d.min(dim=1).values.mean()
    true_to_pred = d.min(dim=0).values.mean()
    return pred_to_true + true_to_pred
