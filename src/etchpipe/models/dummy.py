"""C3: 더미 모델 2종(W2-05) — 파이프라인 골격 검증용. 실제 MeshGraphNet/FNO는 4주차(W4).

ZeroDisplacementModel: 변위 0 예측 — champion(정상 모델) 역할.
RandomDisplacementModel: 무작위 변위 — 게이트가 차단해야 하는 "망가진 모델" 역할.
"""

import numpy as np


class ZeroDisplacementModel:
    """항상 변위 0을 예측한다(입력 형상을 그대로 반환) — 더미 champion."""

    def predict(self, pos: np.ndarray) -> np.ndarray:
        return pos.copy()


class RandomDisplacementModel:
    """정규분포 노이즈를 변위로 더한다 — 게이트가 BLOCK해야 하는 더미 모델.
    CLAUDE.md: 무작위 요소는 시드를 받고 기록한다."""

    def __init__(self, scale: float = 1.0, seed: int = 0) -> None:
        self.scale = scale
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def predict(self, pos: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(scale=self.scale, size=pos.shape)
        return pos + noise
