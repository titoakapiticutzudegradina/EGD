#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

COMMANDS: dict[str, str] = {
    "evaluate": "eval.evaluate",
    "error-analysis": "eval.error_analysis",
    "plot-results": "eval.plot_results",
    "build-test-cumulative": "eval.build_test_cumulative_csv",
    "baseline": "eval.baseline",
    "roberta-goemotions-metrics": "eval.eval_metrics_roberta_goemotions",
}


def _models_sys_path() -> str:
    return str(Path(__file__).resolve().parent / "models")


def _run(module_path: str, argv: list[str]) -> int:
    models_path = _models_sys_path()
    if models_path not in sys.path:
        sys.path.insert(0, models_path)

    mod = import_module(module_path)
    if not hasattr(mod, "main"):
        raise SystemExit(f"{module_path} has no main()")

    old_argv = sys.argv[:]
    try:
        sys.argv = [module_path, *argv]
        mod.main()
        return 0
    finally:
        sys.argv = old_argv


def _strip_optional_separator(argv: list[str]) -> list[str]:
    """Allow `run_models.py CMD -- --flags` as well as `run_models.py CMD --flags`."""
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Entrypoint for training/eval utilities (run from Licenta/ or repo root)."
        ),
        epilog=(
            "Examples:\n"
            "  python3 run_models.py evaluate --model bert_goemotions\n"
            "  python3 run_models.py error-analysis --model baseline roberta_goemotions\n"
            "  python3 run_models.py plot-results -m bert_goemotions roberta_goemotions "
            "-o comparison.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "cmd",
        choices=sorted(COMMANDS),
        help="Which script to run.",
    )
    p.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="Flags passed through to the underlying script.",
    )
    args = p.parse_args()
    forward = _strip_optional_separator(args.forward_args)
    return _run(COMMANDS[args.cmd], forward)


if __name__ == "__main__":
    raise SystemExit(main())
