from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap  

import argparse

import pandas as pd

from core.paths import LICENTA_ROOT
from core.window_sampling import CUMULATIVE_STRATEGY, filter_cumulative_windows
from utils.logger import get_logger


DEFAULT_SRC = LICENTA_ROOT / "data/processed/test_windows.csv"
DEFAULT_OUT = LICENTA_ROOT / "data/processed/test_windows_cumulative.csv"

logger = get_logger("build_test_csv", "logs/build_test_csv.log")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write test_windows_cumulative.csv (window_strategy=full only) "
            "for faster evaluate.py loading."
        )
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main():
    args = parse_args()
    src = args.src
    out = args.out
    first = True
    total_in = 0
    total_out = 0

    logger.info(f"Filtering {src} -> {out} (strategy={CUMULATIVE_STRATEGY})")

    for chunk in pd.read_csv(src, chunksize=args.chunksize):
        total_in += len(chunk)
        filtered = filter_cumulative_windows(chunk)
        total_out += len(filtered)
        if filtered.empty:
            continue
        filtered.to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False

    logger.info(
        f"Done. Wrote {total_out} cumulative rows "
        f"(dropped {total_in - total_out} fixed-window rows)."
    )


if __name__ == "__main__":
    main()
