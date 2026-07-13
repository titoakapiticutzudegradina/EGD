from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


from core.paths import LICENTA_ROOT, TRAIN_PATH

ROOT = LICENTA_ROOT

VAL_SIZE = 0.2
RANDOM_STATE = 42


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        name = "CPU"
    return device, name


def _load_cumulative_train_df(path: Path = TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "window_strategy" in df.columns:
        from core.window_sampling import filter_cumulative_windows

        df = filter_cumulative_windows(df)
    return df


def load_train_val_split(path: Path = TRAIN_PATH):
    df = _load_cumulative_train_df(path)
    return train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )


def load_train_val_split_by_conv(path: Path = TRAIN_PATH):
    df = _load_cumulative_train_df(path)
    if "conv_id" not in df.columns:
        return load_train_val_split(path)

    conv_labels = df.groupby("conv_id", sort=False)["label"].first()
    train_ids, val_ids = train_test_split(
        conv_labels.index.to_numpy(),
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=conv_labels.values,
    )
    train_df = df[df["conv_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["conv_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def predict_probs(model, X, device):
    model.eval()
    return model(X.to(device)).squeeze().cpu().numpy()


@torch.no_grad()
def evaluate(model, X, y, device, threshold=0.5):
    probs = predict_probs(model, X, device)
    binary = (probs >= threshold).astype(int)
    return f1_score(y.numpy(), binary)


def pos_weight_from_labels(y: torch.Tensor) -> float:
    positives = y.sum().item()
    negatives = len(y) - positives
    return negatives / max(positives, 1.0)


def find_best_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    probs = np.asarray(probs).reshape(-1)
    y = np.asarray(y).reshape(-1)
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.01, 0.96, 0.01):
        f1 = f1_score(y, (probs >= threshold).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold, best_f1


def mean_early_detection_f1(
    progress: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    checkpoints: list[float] | None = None,
    *,
    threshold: float | None = None,
) -> float:
    if checkpoints is None:
        from roberta.roberta_config import EARLY_DETECTION_CHECKPOINTS

        checkpoints = EARLY_DETECTION_CHECKPOINTS
    progress = np.asarray(progress, dtype=float).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    y = np.asarray(y).reshape(-1)

    scores: list[float] = []
    for cp in checkpoints:
        mask = progress <= cp
        if not mask.any():
            continue
        if threshold is None:
            _, f1 = find_best_threshold(probs[mask], y[mask])
        else:
            f1 = f1_score(y[mask], (probs[mask] >= threshold).astype(int))
        scores.append(float(f1))
    return float(np.mean(scores)) if scores else 0.0


def find_best_thresholds_by_progress(
    progress: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    checkpoints: list[float] | None = None,
) -> dict[float, float]:
    if checkpoints is None:
        from roberta.roberta_config import EARLY_DETECTION_CHECKPOINTS

        checkpoints = EARLY_DETECTION_CHECKPOINTS
    progress = np.asarray(progress, dtype=float).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    y = np.asarray(y).reshape(-1)

    thresholds: dict[float, float] = {}
    for cp in checkpoints:
        mask = progress <= cp
        if not mask.any():
            continue
        thr, _ = find_best_threshold(probs[mask], y[mask])
        thresholds[float(cp)] = thr
    return thresholds


def make_weighted_bce_loss(pos_weight: float):
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = torch.where(targets > 0.5, pos_weight, 1.0)
        return nn.functional.binary_cross_entropy(preds, targets, weight=w)

    return loss_fn


def make_loader(X, y, batch_size, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
