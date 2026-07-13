from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CUMULATIVE_STRATEGY = "full"


def filter_cumulative_windows(df: pd.DataFrame) -> pd.DataFrame:
    if "window_strategy" not in df.columns:
        return df
    return df.loc[df["window_strategy"] == CUMULATIVE_STRATEGY].copy()


def load_roberta_train_val_split(
    path: Path,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df = filter_cumulative_windows(df)

    if "conv_id" in df.columns:
        conv_labels = df.groupby("conv_id", sort=False)["label"].first()
        train_ids, val_ids = train_test_split(
            conv_labels.index.to_numpy(),
            test_size=val_size,
            random_state=random_state,
            stratify=conv_labels.values,
        )
        train_df = df[df["conv_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["conv_id"].isin(val_ids)].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
            stratify=df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    train_weights = np.ones(len(train_df), dtype=np.float64)
    return train_df, val_df, train_weights
