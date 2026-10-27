"""C6: 레지스트리 최소판(W2-07) — 승격, champion 포인터, history.jsonl.
모델 카드·SHA-256 무결성 검증·롤백은 5주차(W5-07)에서 채운다.
"""

import json
import pathlib
from datetime import UTC, datetime

from etchpipe.gate.gate import GateResult


class Registry:
    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"
        self.current_path = self.root / "current"

    def promote(self, model_id: str, gate_result: GateResult) -> bool:
        """게이트 PASS일 때만 champion 포인터를 바꾼다. 실패해도 history에는 기록한다."""
        entry = {
            "model_id": model_id,
            "promoted": gate_result.passed,
            "exit_code": gate_result.exit_code,
            "reasons": gate_result.reasons,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if gate_result.passed:
            self.current_path.write_text(model_id)
        return gate_result.passed

    def current_champion(self) -> str | None:
        if self.current_path.exists():
            return self.current_path.read_text().strip()
        return None

    def history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text().splitlines() if line]
