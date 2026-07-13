from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap 

import torch

from bert.bert_feature import BERTEmbedder
from goemotions.go_emotions_feature import GoEmotionsEmbedder
from core.training_utils import get_device, load_train_val_split
from utils.logger import get_logger


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache/fused_features.pt"

BERT_EMBED_BATCH = 16
GOEMOTIONS_EMBED_BATCH = 16

logger = get_logger("fuse_features", "logs/fuse_features.log")


def fuse(texts, bert_embedder, go_embedder):
    logger.info("Encoding with DistilBERT...")
    bert_x = bert_embedder.encode(texts, batch_size=BERT_EMBED_BATCH)
    logger.info("Encoding with GoEmotions...")
    go_x = go_embedder.encode(texts, batch_size=GOEMOTIONS_EMBED_BATCH)
    return torch.cat([bert_x, go_x], dim=1)


def main():
    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    train_df, val_df = load_train_val_split()
    logger.info(f"Train size: {len(train_df)}, validation size: {len(val_df)}")

    bert_embedder = BERTEmbedder(device=device)
    go_embedder = GoEmotionsEmbedder(device=device)
    bert_dim = bert_embedder.model.config.hidden_size
    go_dim = go_embedder.num_labels

    train_x = fuse(train_df["text"].tolist(), bert_embedder, go_embedder)
    val_x = fuse(val_df["text"].tolist(), bert_embedder, go_embedder)

    train_y = torch.tensor(train_df["label"].values).float()
    val_y = torch.tensor(val_df["label"].values).float()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_X": train_x,
            "train_y": train_y,
            "val_X": val_x,
            "val_y": val_y,
            "bert_dim": bert_dim,
            "go_dim": go_dim,
        },
        CACHE_PATH,
    )

    logger.info(
        f"Fused features saved to {CACHE_PATH} "
        f"(dim={bert_dim + go_dim}: BERT {bert_dim} + GoEmotions {go_dim})"
    )


if __name__ == "__main__":
    main()
