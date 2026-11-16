"""W6-05: `python -m etchpipe.retrain run` — 전 과정(에스컬레이션 루프 포함) 무개입 실행."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m etchpipe.retrain")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    for command, help_text, intake_help in (
        ("run-batch", "Evaluate one ready observation batch", "Directory with raw/ and graphs/"),
        ("etl-batch", "Convert a raw observation batch, then evaluate it", "Directory with raw/"),
    ):
        batch = sub.add_parser(command, help=help_text)
        batch.add_argument("intake", type=Path, help=intake_help)
        batch.add_argument("--training-raw", type=Path, help="Defaults to <intake>/training")
        batch.add_argument("--registry", type=Path, default=REPO_ROOT / "registry")
        batch.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
        batch.add_argument(
            "--promote", action="store_true", help="Register a gate-passing candidate",
        )
    args = parser.parse_args()

    if args.command == "run":
        from etchpipe.retrain.execute import run

        return run()
    if args.command in {"run-batch", "etl-batch"}:
        from etchpipe.retrain.batch import run_batch

        try:
            if args.command == "etl-batch":
                from etchpipe.retrain.intake import prepare_observation

                prepare_observation(args.intake)
            result = run_batch(
                args.intake, training_raw=args.training_raw or args.intake / "training",
                registry_root=args.registry, runs_root=args.runs_root, promote=args.promote,
            )
        except (OSError, KeyError, ValueError) as exc:
            print(f"Observation batch error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return int(result["status"] in {"error", "champion_changed"})
    return 1


if __name__ == "__main__":
    sys.exit(main())
