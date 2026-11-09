"""C3: 베이스라인 2종(W3-07) — 모델 A/B 학습 전 비교 기준(설계서 §3 C3, 체크리스트 W3-07).

(a) 변위 0: `models/dummy.py`의 `ZeroDisplacementModel`을 그대로 재사용 — 초기 표면(t=0)을
    변화 없이 최종 예측으로 쓴다.
(b) `DepthMlpExtrapolationModel`: 레시피 Θ -> 최종 식각 깊이를 작은 MLP(torch)로 회귀하고,
    초기 표면(t=0)에서 y<0인 점(식각 영역)만 예측 깊이만큼 수직으로 이동(직선 외삽)해
    최종 표면을 만든다. 테이퍼·보잉 등 형상 변화는 의도적으로 모델링하지 않는다 — "단순
    베이스라인"(정의서 §3)이 목적.
"""

import numpy as np
import torch
from torch import nn


class DepthMlp(nn.Module):
    def __init__(self, n_in: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DepthMlpExtrapolationModel:
    """W3-07 베이스라인(b): 레시피->깊이 MLP + 직선 외삽(수직 이동)."""

    def __init__(self, theta_keys: list[str], hidden: int = 16, seed: int = 0) -> None:
        self.theta_keys = theta_keys
        torch.manual_seed(seed)
        self.net = DepthMlp(n_in=len(theta_keys), hidden=hidden)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _theta_vec(self, params: dict) -> np.ndarray:
        return np.array([params[k] for k in self.theta_keys], dtype=np.float64)

    def fit(
        self, thetas: list[dict], final_depths: list[float], epochs: int = 500, lr: float = 0.05
    ) -> None:
        """train 레시피(Θ)->최종 깊이(µm) 회귀. 입력은 평균·표준편차로 정규화(train 기준,
        FR-05와 동일 원칙 — 정규화 통계는 train에서만 계산)."""
        x = np.stack([self._theta_vec(t) for t in thetas])
        self._mean, self._std = x.mean(0), x.std(0) + 1e-8
        xn = (x - self._mean) / self._std
        xt = torch.tensor(xn, dtype=torch.float32)
        yt = torch.tensor(final_depths, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xt), yt)
            loss.backward()
            opt.step()

    def predict_depth(self, params: dict) -> float:
        if self._mean is None or self._std is None:
            msg = "fit()을 먼저 호출해야 한다"
            raise RuntimeError(msg)
        x = (self._theta_vec(params) - self._mean) / self._std
        self.net.eval()
        with torch.no_grad():
            return float(self.net(torch.tensor(x[None], dtype=torch.float32))[0])

    def predict(self, pos0: np.ndarray, params: dict) -> np.ndarray:
        """초기 표면(t=0) pos0에서 y<0인 점(식각 영역)만 예측 깊이만큼 수직 이동.
        음수 깊이 예측은 0으로 클램프(물리적으로 깊이가 음수일 수 없음)."""
        depth_pred = max(self.predict_depth(params), 0.0)
        out = pos0.copy()
        below = out[:, 1] < 0
        out[below, 1] -= depth_pred
        return out
