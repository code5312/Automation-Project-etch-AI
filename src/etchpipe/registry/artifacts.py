"""Locate and authenticate the files belonging to a registered model."""

from pathlib import Path

from etchpipe.registry.registry import sha256_file

FNO_BUNDLE_FILES = ("model.safetensors", "config.json", "inference.py", "preprocess.py")


def model_family(card: dict) -> str:
    family = (card.get("config") or {}).get("model_family", "mgn")
    if family not in ("mgn", "fno"):
        raise ValueError(f"Unsupported model family: {family!r}")
    return family


def resolve_artifact(model_id: str, card: dict, runs_root: Path) -> Path:
    if model_id in ("", ".", "..") or "/" in model_id or "\\" in model_id:
        raise ValueError(f"Invalid model ID: {model_id!r}")
    if model_family(card) == "fno":
        return runs_root / "retrain" / model_id / "model.safetensors"
    if model_id == "mgn_tuned_ep2":
        return runs_root / "mgn_tuned" / "model.pt"
    return runs_root / "retrain" / model_id / "model.pt"


def verify_bundle(registry, model_id: str, card: dict, artifact: Path) -> None:
    if not artifact.is_file() or not registry.verify_artifact(model_id, artifact):
        raise ValueError(f"아티팩트 무결성 검증 실패: {model_id}")
    if model_family(card) == "fno":
        verify_fno_files(model_id, card, artifact)


def verify_fno_files(model_id: str, card: dict, artifact: Path) -> None:
    signatures = card["config"].get("artifact_files_sha256")
    if (
        not isinstance(signatures, dict)
        or set(signatures) != set(FNO_BUNDLE_FILES)
        or artifact.name != FNO_BUNDLE_FILES[0]
        or signatures[artifact.name] != card["artifact_sha256"]
    ):
        raise ValueError(f"Invalid FNO bundle manifest: {model_id}")
    for name in FNO_BUNDLE_FILES[1:]:
        path = artifact.with_name(name)
        if not path.is_file() or sha256_file(path) != signatures[name]:
            raise ValueError(f"FNO 파일 무결성 검증 실패: {model_id}/{name}")
