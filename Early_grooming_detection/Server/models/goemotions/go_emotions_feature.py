from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from .go_emotions_cache import GoEmotionsFeatureCache


class GoEmotionsEmbedder:
    DEFAULT_MODEL = "SamLowe/roberta-base-go_emotions"

    def __init__(
        self,
        model_name=None,
        device=None,
        max_length=128,
        *,
        use_cache=True,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_name = model_name or self.DEFAULT_MODEL
        self.model_name = model_name
        self.max_length = max_length
        self.use_cache = use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        self.num_labels = self.model.config.num_labels
        self._cache: GoEmotionsFeatureCache | None = None

    def _ensure_cache(self) -> GoEmotionsFeatureCache:
        if self._cache is None:
            cache = GoEmotionsFeatureCache(self.model_name, self.max_length)
            cache.load()
            self._cache = cache
        return self._cache

    def _encode_raw(self, texts, batch_size=16):
        features = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.sigmoid(logits)
            features.append(probs.cpu())

        if not features:
            return torch.empty((0, self.num_labels), dtype=torch.float32)
        return torch.cat(features)

    def encode(self, texts, batch_size=16):
        if not texts:
            return torch.empty((0, self.num_labels), dtype=torch.float32)

        if not self.use_cache:
            return self._encode_raw(texts, batch_size=batch_size)

        cache = self._ensure_cache()

        missing = cache.missing_texts(texts)
        stale = cache.stale_texts(texts)
        to_encode = list(dict.fromkeys(missing + stale))
        if to_encode:
            if stale:
                cache.forget_texts(stale)
            cache.add(to_encode, self._encode_raw(to_encode, batch_size=batch_size))
            cache.save()

        return cache.get_batch(texts)
