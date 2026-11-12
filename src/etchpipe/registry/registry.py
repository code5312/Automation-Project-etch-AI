"""C6: 레지스트리(W5-07 완성판) — W2-07 최소판(승격, champion 포인터, history.jsonl)에
자동 모델 카드(`card.json`)·`valid_range.json`·`manifest.json`·SHA-256 무결성·롤백 추가.

설계서 §10: "PASS 판정이 아닌 모델은 `registry/`에 쓰이지 않는다(쓰기 권한은 게이트
단계만 가진다)" — BLOCK된 모델은 카드·아티팩트 디렉터리를 전혀 안 만든다. 단 이건
히스토리 로그(`history.jsonl`)까지 막는 뜻은 아니다: W2-07부터 이미 실패한 시도도
`history.jsonl`에 남겨왔고(`test_promote_on_fail_does_not_set_champion`), W5-06 부정
테스트·W6-06 "시도마다 게이트 결과 기록"이 이 감사 로그에 의존한다.
"""

import hashlib
import json
import pathlib
from datetime import UTC, datetime
from typing import Any

from etchpipe.gate.gate import GateResult


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """아티팩트 무결성(설계서 §3 C6) — 큰 체크포인트도 청크 단위로 읽어 메모리에 한
    번에 안 올린다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_valid_range(
    raw_dir: str | pathlib.Path, run_ids: list[str]
) -> dict[str, dict[str, float]]:
    """`run_ids`(보통 train split)의 실제 Θ 범위(min/max) [제안] — 정의서 §7.1 표의
    샘플링 상한이 아니라 "실제로 학습에 쓰인 범위"를 기록한다(Sobol 샘플링이라 표의
    경계값을 정확히 찍지 않을 수 있음). 서빙(W6-08) `in_range` 판정 기준이 된다."""
    raw_dir = pathlib.Path(raw_dir)
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for rid in run_ids:
        params = json.loads((raw_dir / rid / "meta.json").read_text())["params"]
        for k, v in params.items():
            mins[k] = v if k not in mins else min(mins[k], v)
            maxs[k] = v if k not in maxs else max(maxs[k], v)
    return {k: {"min": mins[k], "max": maxs[k]} for k in mins}


class Registry:
    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"
        self.current_path = self.root / "current"
        self.manifest_path = self.root / "manifest.json"

    def _append_history(self, entry: dict) -> None:
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def promote(self, model_id: str, gate_result: GateResult) -> bool:
        """게이트 PASS일 때만 champion 포인터를 바꾼다. 실패해도 history에는 기록한다."""
        entry = {
            "action": "promote",
            "model_id": model_id,
            "promoted": gate_result.passed,
            "exit_code": gate_result.exit_code,
            "reasons": gate_result.reasons,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._append_history(entry)
        if gate_result.passed:
            self.current_path.write_text(model_id)
        self._write_manifest()
        return gate_result.passed

    def register(
        self,
        model_id: str,
        artifact_path: str | pathlib.Path,
        gate_result: GateResult,
        *,
        dataset_hash: str,
        config: dict[str, Any],
        git_sha: str,
        metrics: dict[str, Any],
        valid_range: dict[str, dict[str, float]],
    ) -> bool:
        """설계서 §3 C6: 카드(`card.json`)+`valid_range.json`을 쓰고 승격 여부를 결정한다.
        BLOCK이면(§10) 카드를 전혀 안 쓰고 `promote`로 history 기록만 남긴다."""
        if gate_result.passed:
            model_dir = self.root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            card = {
                "model_id": model_id,
                "dataset_hash": dataset_hash,
                "config": config,
                "git_sha": git_sha,
                "metrics": metrics,
                "valid_range": valid_range,
                "gate_result": {
                    "passed": gate_result.passed,
                    "exit_code": gate_result.exit_code,
                    "reasons": gate_result.reasons,
                },
                "artifact_sha256": sha256_file(artifact_path),
                "created_at": datetime.now(UTC).isoformat(),
            }
            (model_dir / "card.json").write_text(
                json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (model_dir / "valid_range.json").write_text(
                json.dumps(valid_range, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return self.promote(model_id, gate_result)

    def rollback(self, model_id: str) -> None:
        """champion을 과거에 `register`로 등록된(= `card.json`이 있는) model_id로
        되돌린다 — 게이트 재실행 없는 수동 복구 동작이라 `GateResult`를 안 받는다."""
        if not (self.root / model_id / "card.json").exists():
            raise ValueError(f"등록된 적 없는(card.json 없음) model_id: {model_id!r}")
        self._append_history(
            {
                "action": "rollback",
                "model_id": model_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.current_path.write_text(model_id)
        self._write_manifest()

    def current_champion(self) -> str | None:
        if self.current_path.exists():
            return self.current_path.read_text().strip()
        return None

    def history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text().splitlines() if line]

    def load_card(self, model_id: str) -> dict:
        return json.loads((self.root / model_id / "card.json").read_text(encoding="utf-8"))

    def verify_artifact(self, model_id: str, artifact_path: str | pathlib.Path) -> bool:
        """서빙 로드 시 무결성 검증(설계서 §3 C6) — 카드의 sha256과 실제 파일을 대조."""
        return sha256_file(artifact_path) == self.load_card(model_id)["artifact_sha256"]

    def _write_manifest(self) -> None:
        models = sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "card.json").exists()
        )
        manifest = {
            "current": self.current_champion(),
            "models": models,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
