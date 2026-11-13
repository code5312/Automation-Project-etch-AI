"""W6-05: `python -m etchpipe.retrain run` — 전 과정(에스컬레이션 루프 포함) 무개입 실행."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m etchpipe.retrain")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    args = parser.parse_args()

    if args.command == "run":
        from etchpipe.retrain.execute import run

        return run()
    return 1


if __name__ == "__main__":
    sys.exit(main())
