from __future__ import annotations

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1]
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import bootstrap  

import numpy as np
import torch

from core.fused_classifier import (
    GatedFusionClassifier,
    freeze_roberta_fc,
    init_roberta_from_encoder,
    trainable_parameters,
)
from roberta.roberta_config import EARLY_DETECTION_CHECKPOINTS, ROBERTA_MAX_LENGTH
from core.paths import TRAINED_DIR
from core.training_utils import (
    find_best_threshold,
    find_best_thresholds_by_progress,
    get_device,
    make_loader,
    make_weighted_bce_loss,
    mean_early_detection_f1,
    pos_weight_from_labels,
    train_epoch,
)
from utils.logger import get_logger


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache/roberta_fused_features.pt"
ROBERTA_MODEL_PATH = TRAINED_DIR / "roberta_model.pt"
MODEL_PATH = TRAINED_DIR / "roberta_goemotions_model.pt"

EPOCHS = 15 #number of full passes through the training data
BATCH_SIZE = 32 #number of embeddings in a batch
LR = 5e-3 #learning rate for the optimizer
PATIENCE = 4 #number of epochs without improvement before early stopping

logger = get_logger("roberta_fused_train", "logs/roberta_fused_train.log")

#adapter so training utils can call model(X) on cached concatenated features
class ConcatFusionWrapper(torch.nn.Module):

    def __init__(self, inner: GatedFusionClassifier):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_from_concat(x)


@torch.no_grad()
def predict_probs_wrapped(model, X, device):
    model.eval()
    return model(X.to(device)).squeeze().cpu().numpy()


def save_checkpoint(
    model,
    threshold,
    progress_thresholds,
    roberta_dim,
    go_dim,
    meta_dim,
    path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "progress_thresholds": progress_thresholds,
            "roberta_dim": roberta_dim,
            "go_dim": go_dim,
            "meta_dim": meta_dim,
            "roberta_max_length": ROBERTA_MAX_LENGTH,
            "go_max_length": ROBERTA_MAX_LENGTH,
            "model_type": GatedFusionClassifier.MODEL_TYPE,
        },
        path,
    )


def main():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Fused features not found at {CACHE_PATH}. Run fuse_roberta_features.py first."
        )
    if not ROBERTA_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained RoBERTa model not found at {ROBERTA_MODEL_PATH}. "
            "Run train_roberta.py first."
        )

    device, device_name = get_device()
    logger.info(f"Using device: {device} - {device_name}")

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    cached_ml = cache.get("roberta_max_length")
    if cached_ml is not None and int(cached_ml) != ROBERTA_MAX_LENGTH:
        raise ValueError(
            f"Cache was built with max_length={cached_ml}, "
            f"but ROBERTA_MAX_LENGTH={ROBERTA_MAX_LENGTH}. Re-run fuse_roberta_features.py."
        )

    train_x = cache["train_X"]
    val_x = cache["val_X"]
    train_y = cache["train_y"]
    val_y = cache["val_y"]
    val_progress = np.asarray(
        cache.get("val_progress", []), dtype=float
    ).reshape(-1)
    if val_progress.size != len(val_y):
        raise ValueError(
            "Cache missing val_progress; re-run fuse_roberta_features.py."
        )

    roberta_dim = int(cache["roberta_dim"])
    go_dim = int(cache["go_dim"])
    meta_dim = int(cache.get("meta_dim", 0))
    if meta_dim <= 0:
        raise ValueError(
            "Cache missing meta_dim; re-run fuse_roberta_features.py with metadata."
        )

    expected_dim = roberta_dim + go_dim + meta_dim
    if train_x.shape[1] != expected_dim:
        raise ValueError(
            f"Cache feature dim {train_x.shape[1]} != expected {expected_dim}. "
            "Re-run fuse_roberta_features.py."
        )

    logger.info(
        f"Loaded fused cache: train={len(train_x)}, val={len(val_x)}, "
        f"dim={expected_dim} (RoBERTa {roberta_dim}, Go {go_dim}, meta {meta_dim})"
    )

    core = GatedFusionClassifier(
        roberta_dim=roberta_dim,
        go_dim=go_dim,
        meta_dim=meta_dim,
    ).to(device)
    init_roberta_from_encoder(core, ROBERTA_MODEL_PATH, device)
    freeze_roberta_fc(core)
    model = ConcatFusionWrapper(core).to(device)
    logger.info(
        f"Warm-started RoBERTa logit from {ROBERTA_MODEL_PATH}; "
        f"RoBERTa frozen, GoEmotions refiner + gate trainable (lr={LR})"
    )

    pos_weight = pos_weight_from_labels(train_y)
    logger.info(f"Training with pos_weight={pos_weight:.4f}")
    loss_fn = make_weighted_bce_loss(pos_weight)

    optimizer = torch.optim.Adam(trainable_parameters(core), lr=LR)
    train_loader = make_loader(train_x, train_y, BATCH_SIZE)

    val_probs = predict_probs_wrapped(model, val_x, device)
    baseline_threshold, baseline_f1 = find_best_threshold(val_probs, val_y.numpy())
    baseline_early_f1 = mean_early_detection_f1(
        val_progress, val_probs, val_y.numpy()
    )
    logger.info(
        f"Validation before fused tuning: F1={baseline_f1:.4f} "
        f"(threshold={baseline_threshold:.2f}), "
        f"mean early-detection F1={baseline_early_f1:.4f}"
    )

    best_early_f1 = baseline_early_f1
    best_val_f1 = baseline_f1
    best_threshold = baseline_threshold
    best_progress_thresholds = find_best_thresholds_by_progress(
        val_progress, val_probs, val_y.numpy(), EARLY_DETECTION_CHECKPOINTS
    )
    best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
    epochs_without_gain = 0

    logger.info("Training gated fusion head (GoEmotions refiner + metadata gate)...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_probs = predict_probs_wrapped(model, val_x, device)
        threshold, val_f1 = find_best_threshold(val_probs, val_y.numpy())
        early_f1 = mean_early_detection_f1(
            val_progress, val_probs, val_y.numpy()
        )
        progress_thresholds = find_best_thresholds_by_progress(
            val_progress, val_probs, val_y.numpy(), EARLY_DETECTION_CHECKPOINTS
        )
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}, "
            f"val F1: {val_f1:.4f} (threshold={threshold:.2f}), "
            f"mean early F1: {early_f1:.4f}"
        )
        if early_f1 > best_early_f1 + 1e-6:
            best_early_f1 = early_f1
            best_val_f1 = val_f1
            best_threshold = threshold
            best_progress_thresholds = progress_thresholds
            best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                logger.info(
                    f"Early stopping after {epoch + 1} epochs "
                    f"(no mean early-detection F1 improvement for {PATIENCE} epochs)"
                )
                break

    core.load_state_dict(best_state)
    save_checkpoint(
        core,
        best_threshold,
        best_progress_thresholds,
        roberta_dim,
        go_dim,
        meta_dim,
        MODEL_PATH,
    )
    logger.info(
        f"Best model saved to {MODEL_PATH} "
        f"(val F1={best_val_f1:.4f}, mean early F1={best_early_f1:.4f}, "
        f"threshold={best_threshold:.2f})"
    )
    logger.info(f"Progress thresholds: {best_progress_thresholds}")


if __name__ == "__main__":
    main()
