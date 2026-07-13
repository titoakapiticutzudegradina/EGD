from pathlib import Path

import torch
import torch.nn as nn

from core.classifier import Classifier
from .go_emotions_feature import GoEmotionsEmbedder
from core.training_utils import (
    get_device,
    load_train_val_split,
    make_loader,
    train_epoch,
    evaluate,
)
from utils.logger import get_logger


MODEL_PATH = Path(__file__).resolve().parent.parent / "trained/goemotions_model.pt"

EPOCHS = 3 #number of full passes through the training data
BATCH_SIZE = 32 #number of embeddings in a batch
EMBED_BATCH = 16 #number of text strings in a batch for embedding
LR = 1e-3 #learning rate for the optimizer

logger = get_logger("goemotions_train", "logs/goemotions_train.log")


def main():
    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    train_df, val_df = load_train_val_split()
    logger.info(f"Train size: {len(train_df)}, validation size: {len(val_df)}")

    embedder = GoEmotionsEmbedder(device=device)
    logger.info("Extracting GoEmotions features...")
    train_x = embedder.encode(train_df["text"].tolist(), batch_size=EMBED_BATCH)
    val_x = embedder.encode(val_df["text"].tolist(), batch_size=EMBED_BATCH)

    train_y = torch.tensor(train_df["label"].values).float()
    val_y = torch.tensor(val_df["label"].values).float()

    model = Classifier(input_dim=embedder.num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    train_loader = make_loader(train_x, train_y, BATCH_SIZE)

    logger.info("Training GoEmotions classifier...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_f1 = evaluate(model, val_x, val_y, device)
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}, val F1: {val_f1:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
