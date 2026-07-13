from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim: int, bert_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.bert_dim = bert_dim
        self.fc = nn.Linear(input_dim, 1)

    def _normalize_blocks(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        bert_x = F.normalize(x[:, : self.bert_dim], p=2, dim=1, eps=eps)
        go_x = F.normalize(x[:, self.bert_dim :], p=2, dim=1, eps=eps)
        return torch.cat([bert_x, go_x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bert_dim is not None:
            x = self._normalize_blocks(x)
        return torch.sigmoid(self.fc(x))


def init_fused_from_encoder(
    model: Classifier,
    encoder_dim: int,
    encoder_path: Path,
    device: torch.device,
) -> None:
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    with torch.no_grad():
        model.fc.weight[:, :encoder_dim].copy_(state["fc.weight"])
        model.fc.weight[:, encoder_dim:].zero_()
        model.fc.bias.copy_(state["fc.bias"])


def init_fused_from_bert(
    model: Classifier, bert_dim: int, bert_path: Path, device: torch.device
) -> None:
    init_fused_from_encoder(model, bert_dim, bert_path, device)


def freeze_encoder_columns(model: Classifier, encoder_dim: int) -> None:
    def zero_encoder_grad(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:, :encoder_dim] = 0
        return grad

    model.fc.weight.register_hook(zero_encoder_grad)
