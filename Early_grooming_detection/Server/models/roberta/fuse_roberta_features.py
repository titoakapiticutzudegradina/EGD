from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap 

import torch

from core.metadata_features import MetadataFeatures, batch_metadata_features
from goemotions.go_emotions_feature import GoEmotionsEmbedder
from roberta.roberta_config import ROBERTA_MAX_LENGTH, ROBERTA_MODEL_NAME
from roberta.roberta_feature import RoBERTaEmbedder
from core.training_utils import get_device, load_train_val_split_by_conv
from utils.logger import get_logger


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache/roberta_fused_features.pt"

#number of text strings in a batch for embedding
ROBERTA_EMBED_BATCH = 16 
GOEMOTIONS_EMBED_BATCH = 16 

logger = get_logger("fuse_roberta_features", "logs/fuse_roberta_features.log")

#concatenate RoBERTa, GoEmotions, and metadata features
def fuse(texts, progress, roberta_embedder, go_embedder):
    logger.info("Encoding with RoBERTa...")
    roberta_x = roberta_embedder.encode(texts, batch_size=ROBERTA_EMBED_BATCH)
    logger.info("Encoding with GoEmotions...")
    go_x = go_embedder.encode(texts, batch_size=GOEMOTIONS_EMBED_BATCH)
    logger.info("Extracting metadata features...")
    meta_x = torch.tensor(
        batch_metadata_features(texts, progress=progress), dtype=torch.float32
    )
    return torch.cat([roberta_x, go_x, meta_x], dim=1)


def main():
    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    train_df, val_df = load_train_val_split_by_conv()
    logger.info(f"Train size: {len(train_df)}, validation size: {len(val_df)}")

    roberta_embedder = RoBERTaEmbedder(
        model_name=ROBERTA_MODEL_NAME,
        device=device,
        max_length=ROBERTA_MAX_LENGTH,
    )
    go_embedder = GoEmotionsEmbedder(
        device=device,
        max_length=ROBERTA_MAX_LENGTH,
    )
    roberta_dim = roberta_embedder.model.config.hidden_size
    go_dim = go_embedder.num_labels
    meta_dim = MetadataFeatures.DIM

    logger.info(
        f"RoBERTa max_length={ROBERTA_MAX_LENGTH}, "
        f"GoEmotions max_length={ROBERTA_MAX_LENGTH}"
    )

    train_progress = train_df["progress"].astype(float).tolist()
    val_progress = val_df["progress"].astype(float).tolist()

    train_x = fuse(
        train_df["text"].tolist(),
        train_progress,
        roberta_embedder,
        go_embedder,
    )
    val_x = fuse(
        val_df["text"].tolist(),
        val_progress,
        roberta_embedder,
        go_embedder,
    )

    train_y = torch.tensor(train_df["label"].values).float()
    val_y = torch.tensor(val_df["label"].values).float()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_X": train_x,
            "train_y": train_y,
            "val_X": val_x,
            "val_y": val_y,
            "val_progress": val_progress,
            "roberta_dim": roberta_dim,
            "go_dim": go_dim,
            "meta_dim": meta_dim,
            "roberta_max_length": ROBERTA_MAX_LENGTH,
            "go_max_length": ROBERTA_MAX_LENGTH,
            "model_type": "gated_fusion_mlp",
        },
        CACHE_PATH,
    )

    logger.info(
        f"Fused features saved to {CACHE_PATH} "
        f"(dim={roberta_dim + go_dim + meta_dim}: "
        f"RoBERTa {roberta_dim} + GoEmotions {go_dim} + meta {meta_dim})"
    )


if __name__ == "__main__":
    main()
