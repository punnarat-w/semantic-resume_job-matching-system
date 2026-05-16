from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("lstm_mlp")
class LSTMMLPBinaryMatcher(ResumeRequirementModel):
    """LSTM baseline adapted from the screenshot for binary matching."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dense_dim: int = 64,
        dropout: float = 0.0,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hidden_dim, dense_dim)
        self.classifier = nn.Linear(dense_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (hidden_state, _) = self.lstm(embedded)
        features = F.relu(self.hidden(self.dropout(hidden_state[-1])))
        return self.classifier(features).squeeze(-1)
