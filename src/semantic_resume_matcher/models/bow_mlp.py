from __future__ import annotations

import torch
from torch import nn

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("bow_mlp")
class BagOfWordsMLPMatcher(ResumeRequirementModel):
    """Small example model showing how teammates can plug in alternatives."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.pad_id).unsqueeze(-1)
        embeddings = self.embedding(input_ids) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = embeddings.sum(dim=1) / lengths
        return self.classifier(pooled).squeeze(-1)

