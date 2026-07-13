from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap 

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from bert.bert_feature import BERTEmbedder
from core.classifier import Classifier
from core.training_utils import (
    find_best_threshold,
    find_best_thresholds_by_progress,
    get_device,
    load_train_val_split_by_conv,
    make_weighted_bce_loss,
    pos_weight_from_labels,
    train_epoch,
)
from roberta.roberta_config import EARLY_DETECTION_CHECKPOINTS
from utils.logger import get_logger


from core.paths import TRAINED_DIR

MODEL_PATH = TRAINED_DIR / "bert_model.pt"

EPOCHS = 8
BATCH_SIZE = 32
EMBED_BATCH = 16
LR = 1e-3

logger = get_logger("bert_train", "logs/bert_train.log")


@torch.no_grad()
def evaluate(model, X, y, device, threshold=0.5):
    model.eval()
    preds = model(X.to(device)).squeeze()
    binary = (preds >= threshold).cpu().numpy().astype(int)
    return f1_score(y.numpy(), binary)


def main():
    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    train_df, val_df = load_train_val_split_by_conv()
    logger.info(f"Train size: {len(train_df)}, validation size: {len(val_df)}")

    embedder = BERTEmbedder(device=device)

    logger.info("Generating DistilBERT embeddings...")
    train_x = embedder.encode(train_df["text"].tolist(), batch_size=EMBED_BATCH)
    val_x = embedder.encode(val_df["text"].tolist(), batch_size=EMBED_BATCH)

    train_y = torch.tensor(train_df["label"].values).float()
    val_y = torch.tensor(val_df["label"].values).float()

    pos_weight = pos_weight_from_labels(train_y)
    logger.info(f"Training with pos_weight={pos_weight:.4f}")
    loss_fn = make_weighted_bce_loss(pos_weight)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = Classifier(input_dim=train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    logger.info("Training...")
    best_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    best_progress_thresholds: dict[float, float] = {}
    val_progress = val_df["progress"].astype(float).values

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_probs = model(val_x.to(device)).squeeze().detach().cpu().numpy()
        thr, val_f1 = find_best_threshold(val_probs, val_y.numpy())
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}, "
            f"val F1: {val_f1:.4f} (threshold={thr:.2f})"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = thr
            best_progress_thresholds = find_best_thresholds_by_progress(
                val_progress, val_probs, val_y.numpy(), EARLY_DETECTION_CHECKPOINTS
            )
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": best_threshold,
            "progress_thresholds": best_progress_thresholds,
        },
        MODEL_PATH,
    )
    logger.info(
        f"Model saved to {MODEL_PATH} "
        f"(best val F1={best_f1:.4f}, threshold={best_threshold:.2f})"
    )


if __name__ == "__main__":
    main()
