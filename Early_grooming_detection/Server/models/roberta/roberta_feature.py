from transformers import AutoModel, AutoTokenizer
import torch


from .roberta_config import ROBERTA_MAX_LENGTH, ROBERTA_MODEL_NAME


class RoBERTaEmbedder:

    def __init__(
        self,
        model_name: str | None = None,
        device=None,
        max_length: int | None = None,
    ):
        model_name = model_name or ROBERTA_MODEL_NAME
        max_length = ROBERTA_MAX_LENGTH if max_length is None else max_length
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=16):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,#list of text strings to tokenize
                padding=True,#pad shorter sequences to max_length
                truncation=True,#truncate longer sequences to max_length
                max_length=self.max_length,
                return_tensors="pt",#return pytorch tensors
            ).to(self.device)
            #no gradient computation
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu())

        return torch.cat(embeddings)
