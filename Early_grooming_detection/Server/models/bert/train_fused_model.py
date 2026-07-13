from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap 

import torch
import torch.nn as nn

from core.classifier import Classifier, freeze_encoder_columns, init_fused_from_bert
from core.paths import TRAINED_DIR
from core.training_utils import (
    find_best_threshold,
    get_device,
    make_loader,
    predict_probs,
    train_epoch,
)
from utils.logger import get_logger


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache/fused_features.pt"
BERT_MODEL_PATH = TRAINED_DIR / "bert_model.pt"
MODEL_PATH = TRAINED_DIR / "bert_goemotions_model.pt"

EPOCHS = 6
BATCH_SIZE = 32
LR = 5e-3

logger = get_logger("fused_model_train", "logs/fused_model_train.log")


def save_checkpoint(model, threshold, bert_dim, go_dim, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "bert_dim": bert_dim,
            "go_dim": go_dim,
        },
        path,
    )


def main():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Fused features not found at {CACHE_PATH}. Run fuse_features.py first."
        )
    if not BERT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained BERT model not found at {BERT_MODEL_PATH}. Run train_bert.py first."
        )

    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    train_x = cache["train_X"]
    val_x = cache["val_X"]
    train_y = cache["train_y"]
    val_y = cache["val_y"]
    bert_dim = cache["bert_dim"]
    go_dim = cache["go_dim"]
    input_dim = bert_dim + go_dim

    logger.info(f"Loaded fused cache: train={len(train_x)}, val={len(val_x)}, dim={input_dim}")

    model = Classifier(input_dim=input_dim, bert_dim=bert_dim).to(device)
    init_fused_from_bert(model, bert_dim, BERT_MODEL_PATH, device)
    freeze_encoder_columns(model, bert_dim)
    logger.info(
        f"Initialized from {BERT_MODEL_PATH}; BERT columns frozen, "
        "GoEmotions columns trainable"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    train_loader = make_loader(train_x, train_y, BATCH_SIZE)

    val_probs = predict_probs(model, val_x, device)
    baseline_threshold, baseline_f1 = find_best_threshold(val_probs, val_y.numpy())
    logger.info(
        f"Validation F1 before GoEmotions tuning: {baseline_f1:.4f} "
        f"(threshold={baseline_threshold:.2f})"
    )

    best_val_f1 = baseline_f1
    best_threshold = baseline_threshold
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info("Training GoEmotions correction on fused features...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_probs = predict_probs(model, val_x, device)
        threshold, val_f1 = find_best_threshold(val_probs, val_y.numpy())
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}, "
            f"val F1: {val_f1:.4f} (threshold={threshold:.2f})"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_checkpoint(model, best_threshold, bert_dim, go_dim, MODEL_PATH)
    logger.info(
        f"Best model saved to {MODEL_PATH} "
        f"(val F1={best_val_f1:.4f}, threshold={best_threshold:.2f})"
    )


if __name__ == "__main__":
    main()
