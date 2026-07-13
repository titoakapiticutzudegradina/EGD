from __future__ import annotations

import argparse

import pandas as pd

from .go_emotions_feature import GoEmotionsEmbedder
from .go_emotions_cache import cache_path
from core.training_utils import ROOT, get_device
from utils.logger import get_logger


DEFAULT_TRAIN_PATH = ROOT / "data/processed/train_windows.csv"
DEFAULT_TEST_PATH = ROOT / "data/processed/test_windows.csv"

logger = get_logger("cache_goemotions", "logs/cache_goemotions.log")


def collect_unique_texts(*paths) -> list[str]:
    texts: set[str] = set()
    for path in paths:
        df = pd.read_csv(path)
        texts.update(df["text"].astype(str).tolist())
    return sorted(texts)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute and cache GoEmotions features for training and test texts."
        )
    )
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument(
        "--goemotions-model",
        default=GoEmotionsEmbedder.DEFAULT_MODEL,
        help="GoEmotions model used for feature extraction.",
    )
    parser.add_argument("--goemotions-max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    texts = collect_unique_texts(args.train_path, args.test_path)
    logger.info(
        f"Collected {len(texts)} unique texts from "
        f"{args.train_path} and {args.test_path}"
    )

    embedder = GoEmotionsEmbedder(
        model_name=args.goemotions_model,
        device=device,
        max_length=args.goemotions_max_length,
        use_cache=True,
    )
    embedder.encode(texts, batch_size=args.batch_size)

    out_path = cache_path(args.goemotions_model, args.goemotions_max_length)
    logger.info(f"GoEmotions cache saved to {out_path}")


if __name__ == "__main__":
    main()
