"""C3: 모델 B — FNO 래퍼(W6-11, FR-08). GPU 팀원이 전달한 `runs/fno_tuned/`의 실제
학습된 체크포인트(`inference.py`/`preprocess.py`)를 그대로 감싼다 — **재구현하지
않는다**: SDF 내부 경계 필터링(마스크/Si 접촉면 제거)처럼 미묘하게 틀리기 쉬운 부분이
이미 검증된 상태라, `etl.py`의 자체 SDF 구현과 부호 규약은 같아도 필터링 알고리즘이
달라(걷기 기반 vs y_below 컷오프) 결과가 달라질 수 있다 — 그대로 재사용하는 게 안전
[확인, 2026-11-12: `python runs/fno_tuned/inference.py` 그대로 실행해 README 문서화된
결과(final depth true 33.22 um / pred 33.05 um)와 일치함을 확인].

모델 A(`MeshGraphNetModel`)는 그래프(점+연결)에서 동작하고 모델 B(FNO)는 SDF 격자에서
동작해 입출력 형태가 근본적으로 다르다 — 이 래퍼는 `predict_curve`로 "t=0 .vtp + 레시피
Θ + 스텝 수 -> 최종 표면선 (N,2) µm"을 공통 출력으로 맞춰, W6-12 모델 A/B 비교표에서
같은 Chamfer/깊이 지표로 바로 비교할 수 있게 한다. 실제 학습(GPU, 원래 계획이던
"CPU 학습(격자 128×256)"에서 변경)은 GPU 팀원이 수행 — 여기서는 전달받은 체크포인트를
불러와 추론만 한다.
"""

import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "fno_tuned"


class FNOModel:
    """`run_dir`(기본 `runs/fno_tuned/`)의 `inference.py`/`preprocess.py`/`config.json`/
    `model.safetensors`를 지연 로드한다 — physicsnemo·scikit-image 없는 환경(예: 아직
    이 패키지를 안 설치한 셋업)에서도 이 클래스 자체는 import 가능하게(`eval.load_model_a`
    와 같은 지연 임포트 패턴)."""

    def __init__(
        self, run_dir: str | pathlib.Path = DEFAULT_RUN_DIR, device: str | None = None
    ) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self._device = device
        self._inference = None
        self._preprocess = None
        self._model = None
        self._cfg: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        run_dir_str = str(self.run_dir)
        if run_dir_str not in sys.path:
            sys.path.insert(0, run_dir_str)
        import inference as fno_inference  # type: ignore[import-not-found]
        import preprocess as fno_preprocess  # type: ignore[import-not-found]

        self._inference = fno_inference
        self._preprocess = fno_preprocess
        self._model, self._cfg = fno_inference.load(run_dir_str, device=self._device)

    @property
    def cfg(self) -> dict:
        self._ensure_loaded()
        assert self._cfg is not None
        return self._cfg

    def sdf_from_vtp(self, vtp_path: str | pathlib.Path) -> np.ndarray:
        """t=0(또는 임의 시점) `.vtp` -> 학습 때와 같은 SDF 격자(NY, NX) µm."""
        self._ensure_loaded()
        return self._preprocess.vtp_to_sdf(str(vtp_path), self._cfg)

    def rollout_sdf(self, sdf0: np.ndarray, params: dict, n_steps: int) -> np.ndarray:
        """`inference.rollout` 그대로 — (n_steps+1, NY, NX) µm, `out[0]`=sdf0."""
        self._ensure_loaded()
        return self._inference.rollout(self._model, self._cfg, sdf0, params, n_steps)

    def curve_from_sdf(self, sdf: np.ndarray) -> np.ndarray:
        """`inference.phi_to_curve` 그대로 — SDF 0등고선을 왼쪽→오른쪽 (N,2) µm로."""
        self._ensure_loaded()
        return self._inference.phi_to_curve(sdf, self._cfg)

    def predict_curve(
        self, vtp_path: str | pathlib.Path, params: dict, n_steps: int
    ) -> np.ndarray:
        """편의 함수: t=0 `.vtp` + Θ + 스텝 수 -> 최종 표면선 (N,2) µm(모델 A와 같은
        출력 형태, W6-12 비교용)."""
        sdf0 = self.sdf_from_vtp(vtp_path)
        sdfs = self.rollout_sdf(sdf0, params, n_steps)
        return self.curve_from_sdf(sdfs[-1])


def load_fno_config(run_dir: str | pathlib.Path = DEFAULT_RUN_DIR) -> dict:
    """physicsnemo 없이 `config.json`만 읽을 때(예: `theta_range`/격자 확인용)."""
    return json.loads((pathlib.Path(run_dir) / "config.json").read_text())
