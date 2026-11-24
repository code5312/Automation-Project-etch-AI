"""W6-05: `python -m etchpipe.retrain run` — 전 과정(에스컬레이션 루프 포함) 무개입 실행."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m etchpipe.retrain")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    # ADR-0008: fine-tune 단계만 GPU로 돌릴 수 있게(게이트 평가는 범위 제외, CPU 고정).
    run_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.command == "run":
        from etchpipe.retrain.execute import run

        return run(device=args.device)
    return 1


if __name__ == "__main__":
    sys.exit(main())
