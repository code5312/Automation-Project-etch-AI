"""W6-04: 자동 반복 루프(최대 4회, ADR-0002) 오케스트레이션.

실제 미세조정(에폭 실행)은 **[GPU 팀원]** 담당(체크리스트 W6-04 주석 그대로) — 이
모듈은 시도 순서·게이트 호출·기록만 한다("[CPU/매핑]"). `fine_tune_fn`/`gate_fn`/
`regenerate_fn`은 주입(DI)해서, 테스트에서는 스텁을 쓰고 실제 실행에서는 GPU 팀원의
학습 엔트리포인트·`gate.run_gate`·ViennaPS 추가 호출을 넘긴다.

에스컬레이션 표는 ADR-0002(`docs/adr/0002-auto-retry-retrain-loop.md`)와 동일 — 수치는
[제안](실측 전 가정값, ADR 자체 태그).
"""

from collections.abc import Callable
from typing import Any

# attempt별 설정. regenerate=True면 그 시도 전에 regenerate_fn(n_train)을 먼저 호출한다
# (ADR-0002: 2차 시도에서만 드리프트 train 40→100개로 확장, ViennaPS 추가 호출).
ATTEMPTS: list[dict[str, Any]] = [
    {"attempt": 1, "n_train": 40, "epochs": (3, 5), "lr_mult": 1.0, "regenerate": False},
    {"attempt": 2, "n_train": 100, "epochs": (3, 5), "lr_mult": 1.0, "regenerate": True},
    {"attempt": 3, "n_train": 100, "epochs": (10, 15), "lr_mult": 1.0, "regenerate": False},
    {"attempt": 4, "n_train": 100, "epochs": (10, 15), "lr_mult": 0.5, "regenerate": False},
]


def run_retrain_loop(
    champion_ckpt: Any,
    fine_tune_fn: Callable[[Any, dict], Any],
    gate_fn: Callable[[Any], dict],
    regenerate_fn: Callable[[int], None] | None = None,
    max_attempts: int = 4,
) -> dict:
    """ADR-0002 표 그대로: 시도마다 (필요하면 `regenerate_fn`으로 데이터 확장 →)
    `fine_tune_fn`으로 champion에서 미세조정 → `gate_fn`으로 판정. `gate_fn`이 반환하는
    dict에 `"passed": bool`이 있으면 그 시도에서 PASS인지 판정한다(`gate.GateResult`를
    `dataclasses.asdict`한 것과 같은 모양이면 바로 맞음). PASS면 **즉시 승격하고 루프
    종료**, `max_attempts`(기본 4)회 모두 BLOCK이면 champion 유지로 종료한다 — 어느
    경우든 `attempts`(W6-05 `retrain_<ts>.json`과 같은 키 이름)에 시도별 config·gate
    결과를 전부 남긴다."""
    attempts_log: list[dict[str, Any]] = []
    for cfg in ATTEMPTS[:max_attempts]:
        if cfg["regenerate"] and regenerate_fn is not None:
            regenerate_fn(cfg["n_train"])
        new_ckpt = fine_tune_fn(champion_ckpt, cfg)
        gate_result = gate_fn(new_ckpt)
        attempts_log.append({
            "attempt": cfg["attempt"], "config": cfg, "gate_result": gate_result,
        })
        if gate_result["passed"]:
            return {"promoted": True, "promoted_ckpt": new_ckpt, "attempts": attempts_log}
    return {"promoted": False, "promoted_ckpt": None, "attempts": attempts_log}
