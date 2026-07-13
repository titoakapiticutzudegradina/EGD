from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class GatedFusionClassifier(nn.Module):
    MODEL_TYPE = "gated_fusion_mlp"

    def __init__(
        self,
        roberta_dim: int,
        go_dim: int,
        meta_dim: int,
        *,
        go_hidden: int = 32,
        go_refined_dim: int = 16,
        aux_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.roberta_dim = roberta_dim
        self.go_dim = go_dim
        self.meta_dim = meta_dim
        self.go_refined_dim = go_refined_dim

        self.roberta_fc = nn.Linear(roberta_dim, 1)

        self.go_refiner = nn.Sequential(
            nn.Linear(go_dim, go_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(go_hidden, go_refined_dim),
            nn.ReLU(),
        )

        aux_in = go_refined_dim + meta_dim
        self.gate_net = nn.Sequential(
            nn.Linear(aux_in, aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 1),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(aux_in, aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 1),
        )

        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(
        self,
        roberta_x: torch.Tensor,
        go_x: torch.Tensor,
        meta_x: torch.Tensor,
    ) -> torch.Tensor:
        roberta_logit = self.roberta_fc(roberta_x)
        go_refined = self.go_refiner(go_x)
        aux = torch.cat([go_refined, meta_x], dim=1)
        gate = torch.sigmoid(self.gate_net(aux))
        delta = self.delta_net(aux)
        logit = roberta_logit + gate * delta
        return torch.sigmoid(logit.squeeze(-1))

    def forward_from_concat(self, x: torch.Tensor) -> torch.Tensor:
        end_roberta = self.roberta_dim
        end_go = end_roberta + self.go_dim
        return self.forward(
            x[:, :end_roberta],
            x[:, end_roberta:end_go],
            x[:, end_go:],
        )


def init_roberta_from_encoder(
    model: GatedFusionClassifier,
    encoder_path: Path,
    device: torch.device,
) -> None:
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    with torch.no_grad():
        model.roberta_fc.weight.copy_(state["fc.weight"])
        model.roberta_fc.bias.copy_(state["fc.bias"])


def freeze_roberta_fc(model: GatedFusionClassifier) -> None:
    for param in model.roberta_fc.parameters():
        param.requires_grad = False


def trainable_parameters(model: GatedFusionClassifier):
    return [p for p in model.parameters() if p.requires_grad]
