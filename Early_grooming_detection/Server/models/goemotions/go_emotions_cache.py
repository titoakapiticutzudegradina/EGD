from __future__ import annotations

import re
import threading
from pathlib import Path

import torch

from core.paths import CACHE_DIR

LEGACY_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _safe_model_slug(model_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model_name.replace("/", "--"))


def _cache_filename(model_name: str, max_length: int) -> str:
    slug = _safe_model_slug(model_name)
    return f"goemotions_{slug}_ml{max_length}.pt"


def cache_path(model_name: str, max_length: int) -> Path:
    return CACHE_DIR / _cache_filename(model_name, max_length)


def legacy_cache_path(model_name: str, max_length: int) -> Path:
    return LEGACY_CACHE_DIR / _cache_filename(model_name, max_length)


def _candidate_cache_paths(model_name: str, max_length: int) -> list[Path]:
    primary = cache_path(model_name, max_length)
    legacy = legacy_cache_path(model_name, max_length)
    if legacy == primary:
        return [primary]
    return [primary, legacy]


class GoEmotionsFeatureCache:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self.path = cache_path(model_name, max_length)
        self.texts: list[str] = []
        self.features: torch.Tensor | None = None
        self._index: dict[str, int] = {}
        self._dirty = False
        self._lock = threading.Lock()

    def load(self) -> bool:
        for candidate in _candidate_cache_paths(self.model_name, self.max_length):
            if not candidate.is_file():
                continue

            data = torch.load(candidate, map_location="cpu", weights_only=False)
            if (
                data.get("model_name") != self.model_name
                or data.get("max_length") != self.max_length
            ):
                continue

            self.texts = list(data["texts"])
            self.features = data["features"]
            if self.repair():
                self.save()
            return True

        return False

    def save(self) -> None:
        with self._lock:
            if not self._dirty or self.features is None:
                return
            payload = {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "num_labels": int(self.features.shape[1]),
                "texts": self.texts,
                "features": self.features,
            }
            self._dirty = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.path)

    def _rebuild_index(self) -> None:
        self._index = {}
        for idx, text in enumerate(self.texts):
            if text not in self._index:
                self._index[text] = idx

    def repair(self) -> bool:
        with self._lock:
            return self._repair_unlocked()

    def _repair_unlocked(self) -> bool:
        if self.features is None:
            self._index = {}
            return False

        changed = False
        n_texts = len(self.texts)
        n_features = int(self.features.shape[0])
        if n_texts != n_features:
            n = min(n_texts, n_features)
            self.texts = self.texts[:n]
            self.features = self.features[:n]
            changed = True

        if changed or not self._index:
            self._rebuild_index()

        if changed:
            self._dirty = True
        return changed

    def stale_texts(self, texts: list[str]) -> list[str]:
        with self._lock:
            return self._stale_texts_unlocked(texts)

    def _stale_texts_unlocked(self, texts: list[str]) -> list[str]:
        if self.features is None:
            return []
        n = int(self.features.shape[0])
        stale: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            idx = self._index.get(text)
            if idx is not None and idx >= n:
                stale.append(text)
        return stale

    def forget_texts(self, texts: list[str]) -> None:
        with self._lock:
            for text in texts:
                self._index.pop(text, None)

    def get_batch(self, texts: list[str]) -> torch.Tensor:
        with self._lock:
            return self._get_batch_unlocked(texts)

    def _get_batch_unlocked(self, texts: list[str]) -> torch.Tensor:
        if self.features is None:
            raise RuntimeError("GoEmotions cache is empty.")
        n = int(self.features.shape[0])
        indices = []
        for text in texts:
            idx = self._index.get(text)
            if idx is None or idx >= n:
                raise KeyError(text)
            indices.append(idx)
        return self.features[indices]

    def missing_texts(self, texts: list[str]) -> list[str]:
        with self._lock:
            return list(dict.fromkeys(text for text in texts if text not in self._index))

    def add(self, texts: list[str], features: torch.Tensor) -> None:
        with self._lock:
            self._add_unlocked(texts, features)

    def _add_unlocked(self, texts: list[str], features: torch.Tensor) -> None:
        if not texts:
            return

        features = features.cpu().float()
        new_texts: list[str] = []
        new_rows: list[torch.Tensor] = []
        for text, row in zip(texts, features):
            if text in self._index:
                continue
            new_texts.append(text)
            new_rows.append(row.unsqueeze(0))

        if not new_rows:
            return

        new_block = torch.cat(new_rows, dim=0)
        start_idx = len(self.texts) if self.features is not None else 0
        if self.features is None:
            self.texts = new_texts
            self.features = new_block
        else:
            self.texts.extend(new_texts)
            self.features = torch.cat([self.features, new_block], dim=0)

        for offset, text in enumerate(new_texts):
            self._index[text] = start_idx + offset
        self._dirty = True
