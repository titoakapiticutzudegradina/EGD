from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import torch

from bert.bert_feature import BERTEmbedder
from core.classifier import Classifier
from core.fused_classifier import GatedFusionClassifier
from core.metadata_features import batch_metadata_features
from goemotions.go_emotions_feature import GoEmotionsEmbedder
from roberta.roberta_config import EARLY_DETECTION_CHECKPOINTS, ROBERTA_MAX_LENGTH
from roberta.roberta_feature import RoBERTaEmbedder


from core.paths import EVALUATED_DIR, MODELS_DIR, TRAINED_DIR

ROOT = MODELS_DIR.parent 

CHECKPOINTS = EARLY_DETECTION_CHECKPOINTS


def threshold_for_progress(
    progress: float,
    progress_thresholds: dict[float, float],
    default: float,
) -> float:
    for cp in CHECKPOINTS:
        if progress <= cp:
            return float(
                progress_thresholds.get(float(cp), progress_thresholds.get(cp, default))
            )
    if CHECKPOINTS:
        last = float(CHECKPOINTS[-1])
        return float(
            progress_thresholds.get(last, progress_thresholds.get(CHECKPOINTS[-1], default))
        )
    return default

RESULT_FILES = {
    "baseline": "early_detection_results.csv",
    "bert": "bert_early_detection_results.csv",
    "goemotions": "goemotions_early_detection_results.csv",
    "bert_goemotions": "bert_goemotions_early_detection_results.csv",
    "roberta": "roberta_early_detection_results.csv",
    "roberta_ft": "roberta_ft_early_detection_results.csv",
    "roberta_goemotions": "roberta_goemotions_early_detection_results.csv",
}


class Predictor(ABC):
    name: str

    @abstractmethod
    def predict_proba(
        self, texts: list[str], *, progress: list[float] | None = None
    ) -> np.ndarray:
        """Predatory class probabilities in [0, 1], one per input text."""

    def predict(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = self.predict_proba(texts, progress=progress)
        threshold = getattr(self, "threshold", None)
        if threshold is None:
            threshold = getattr(self, "THRESHOLD", 0.5)
        progress_thresholds = getattr(self, "progress_thresholds", None) or {}
        if progress is not None and progress_thresholds:
            return np.array(
                [
                    int(
                        p
                        >= threshold_for_progress(
                            float(pr), progress_thresholds, float(threshold)
                        )
                    )
                    for p, pr in zip(probs, progress)
                ],
                dtype=int,
            )
        return (probs >= threshold).astype(int)


class BaselinePredictor(Predictor):
    name = "baseline"

    def __init__(self):
        self.model = joblib.load(TRAINED_DIR / "baseline_model.joblib")
        self.vectorizer = joblib.load(TRAINED_DIR / "tfidf_vectorizer.joblib")

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]


class BertPredictor(Predictor):
    name = "bert"

    PRED_BATCH_SIZE = 32
    EMBED_BATCH_SIZE = 16
    THRESHOLD = 0.5

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ckpt_path = TRAINED_DIR / "bert_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            self.threshold = float(ckpt.get("threshold", self.THRESHOLD))
            self.progress_thresholds = ckpt.get("progress_thresholds") or {}
        else:
            state_dict = ckpt
            self.threshold = self.THRESHOLD
            self.progress_thresholds = {}

        self.embedder = BERTEmbedder(device=self.device)
        embed_dim = self.embedder.model.config.hidden_size

        self.model = Classifier(input_dim=embed_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), self.PRED_BATCH_SIZE):
            batch_texts = texts[i : i + self.PRED_BATCH_SIZE]
            X = self.embedder.encode(batch_texts, batch_size=self.EMBED_BATCH_SIZE).to(
                self.device
            )
            with torch.no_grad():
                batch_probs = self.model(X).squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=float)


class GoEmotionsPredictor(Predictor):
    name = "goemotions"

    PRED_BATCH_SIZE = 32
    EMBED_BATCH_SIZE = 16
    THRESHOLD = 0.5

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.embedder = GoEmotionsEmbedder(device=self.device)
        self.model = Classifier(input_dim=self.embedder.num_labels).to(self.device)
        self.model.load_state_dict(
            torch.load(
                TRAINED_DIR / "goemotions_model.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.eval()

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), self.PRED_BATCH_SIZE):
            batch_texts = texts[i : i + self.PRED_BATCH_SIZE]
            X = self.embedder.encode(batch_texts, batch_size=self.EMBED_BATCH_SIZE).to(
                self.device
            )
            with torch.no_grad():
                batch_probs = self.model(X).squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=float)


class BertGoEmotionsPredictor(Predictor):
    name = "bert_goemotions"

    PRED_BATCH_SIZE = 32
    BERT_EMBED_BATCH = 16
    GOEMOTIONS_EMBED_BATCH = 16

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.bert_embedder = BERTEmbedder(device=self.device)
        self.go_embedder = GoEmotionsEmbedder(device=self.device)
        bert_dim = self.bert_embedder.model.config.hidden_size
        go_dim = self.go_embedder.num_labels

        checkpoint = torch.load(
            TRAINED_DIR / "bert_goemotions_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.threshold = checkpoint.get("threshold", 0.5)
            state_dict = checkpoint["state_dict"]
        else:
            self.threshold = 0.5
            state_dict = checkpoint

        self.model = Classifier(
            input_dim=bert_dim + go_dim, bert_dim=bert_dim
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        bert_x = self.bert_embedder.encode(
            texts, batch_size=self.BERT_EMBED_BATCH
        )
        go_x = self.go_embedder.encode(
            texts, batch_size=self.GOEMOTIONS_EMBED_BATCH
        )
        return torch.cat([bert_x, go_x], dim=1)

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), self.PRED_BATCH_SIZE):
            batch_texts = texts[i : i + self.PRED_BATCH_SIZE]
            X = self._encode_batch(batch_texts).to(self.device)
            with torch.no_grad():
                batch_probs = self.model(X).squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=float)


class RobertaPredictor(Predictor):
    name = "roberta"

    PRED_BATCH_SIZE = 32
    EMBED_BATCH_SIZE = 16
    THRESHOLD = 0.5

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        max_length = ROBERTA_MAX_LENGTH
        ckpt_path = TRAINED_DIR / "roberta_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            self.threshold = float(ckpt.get("threshold", self.THRESHOLD))
            max_length = int(ckpt.get("roberta_max_length", max_length))
            self.progress_thresholds = ckpt.get("progress_thresholds") or {}
        else:
            state_dict = ckpt
            self.threshold = self.THRESHOLD
            self.progress_thresholds = {}

        self.embedder = RoBERTaEmbedder(device=self.device, max_length=max_length)
        embed_dim = self.embedder.model.config.hidden_size

        self.model = Classifier(input_dim=embed_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), self.PRED_BATCH_SIZE):
            batch_texts = texts[i : i + self.PRED_BATCH_SIZE]
            X = self.embedder.encode(batch_texts, batch_size=self.EMBED_BATCH_SIZE).to(
                self.device
            )
            with torch.no_grad():
                batch_probs = self.model(X).squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=float)


class RobertaGoEmotionsPredictor(Predictor):
    name = "roberta_goemotions"

    PRED_BATCH_SIZE = 32
    ROBERTA_EMBED_BATCH = 16
    GOEMOTIONS_EMBED_BATCH = 16

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        max_length = ROBERTA_MAX_LENGTH
        go_max_length = ROBERTA_MAX_LENGTH
        checkpoint = torch.load(
            TRAINED_DIR / "roberta_goemotions_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.threshold = float(checkpoint.get("threshold", 0.5))
            state_dict = checkpoint["state_dict"]
            enc_dim = checkpoint.get("roberta_dim")
            go_dim = checkpoint.get("go_dim")
            meta_dim = checkpoint.get("meta_dim", 0)
            model_type = checkpoint.get("model_type", "linear")
            max_length = int(checkpoint.get("roberta_max_length", max_length))
            go_max_length = int(checkpoint.get("go_max_length", go_max_length))
            raw_pt = checkpoint.get("progress_thresholds") or {}
            self.progress_thresholds = {
                float(k): float(v) for k, v in raw_pt.items()
            }
        else:
            self.threshold = 0.5
            state_dict = checkpoint
            enc_dim = None
            go_dim = None
            meta_dim = 0
            model_type = "linear"
            self.progress_thresholds = {}

        self.roberta_embedder = RoBERTaEmbedder(
            device=self.device, max_length=max_length
        )
        self.go_embedder = GoEmotionsEmbedder(
            device=self.device, max_length=go_max_length
        )
        roberta_dim = self.roberta_embedder.model.config.hidden_size
        go_dim = go_dim if go_dim is not None else self.go_embedder.num_labels
        enc_dim = enc_dim if enc_dim is not None else roberta_dim
        self.model_type = model_type
        self.roberta_dim = roberta_dim
        self.go_dim = go_dim
        self.meta_dim = int(meta_dim or 0)

        if model_type == GatedFusionClassifier.MODEL_TYPE:
            if self.meta_dim <= 0:
                raise ValueError(
                    "Gated fusion checkpoint missing meta_dim; re-run fused pipeline."
                )
            self.model = GatedFusionClassifier(
                roberta_dim=roberta_dim,
                go_dim=go_dim,
                meta_dim=self.meta_dim,
            ).to(self.device)
        else:
            self.model = Classifier(
                input_dim=roberta_dim + go_dim, bert_dim=enc_dim
            ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _encode_batch(
        self,
        texts: list[str],
        *,
        progress: list[float] | None = None,
    ) -> torch.Tensor:
        roberta_x = self.roberta_embedder.encode(
            texts, batch_size=self.ROBERTA_EMBED_BATCH
        )
        go_x = self.go_embedder.encode(
            texts, batch_size=self.GOEMOTIONS_EMBED_BATCH
        )
        if self.model_type == GatedFusionClassifier.MODEL_TYPE:
            if progress is None:
                progress = [1.0] * len(texts)
            meta_x = torch.tensor(
                batch_metadata_features(texts, progress=progress),
                dtype=torch.float32,
            )
            return torch.cat([roberta_x, go_x, meta_x], dim=1)
        return torch.cat([roberta_x, go_x], dim=1)

    def predict_proba(self, texts: list[str], *, progress: list[float] | None = None) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), self.PRED_BATCH_SIZE):
            batch_texts = texts[i : i + self.PRED_BATCH_SIZE]
            batch_progress = None
            if progress is not None:
                batch_progress = progress[i : i + self.PRED_BATCH_SIZE]
            X = self._encode_batch(batch_texts, progress=batch_progress).to(
                self.device
            )
            with torch.no_grad():
                if self.model_type == GatedFusionClassifier.MODEL_TYPE:
                    end_r = self.roberta_dim
                    end_g = end_r + self.go_dim
                    batch_probs = self.model(
                        X[:, :end_r],
                        X[:, end_r:end_g],
                        X[:, end_g:],
                    )
                else:
                    batch_probs = self.model(X).squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=float)


PREDICTORS: dict[str, type[Predictor]] = {
    "baseline": BaselinePredictor,
    "bert": BertPredictor,
    "goemotions": GoEmotionsPredictor,
    "bert_goemotions": BertGoEmotionsPredictor,
    "roberta": RobertaPredictor,
    "roberta_goemotions": RobertaGoEmotionsPredictor,
}


def list_models() -> list[str]:
    return sorted(PREDICTORS.keys())


def get_predictor(name: str) -> Predictor:
    if name not in PREDICTORS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {', '.join(list_models())}"
        )
    return PREDICTORS[name]()


def results_path(name: str) -> Path:
    if name not in RESULT_FILES:
        raise ValueError(
            f"No results file configured for {name!r}. "
            f"Add an entry to RESULT_FILES in predictors.py."
        )
    return EVALUATED_DIR / RESULT_FILES[name]
