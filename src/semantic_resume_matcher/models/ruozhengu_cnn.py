from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("ruozhengu_cnn")
class RuoZhenguCNNBinaryMatcher(ResumeRequirementModel):
    """
    Reference: https://github.com/ruozhengu/job-resume-matching-algo/blob/master/cnn/CNN.py
    
    A repo CNN encoder adapted for binary requirement matching.

    The referenced model uses:
    Embedding -> Conv1D -> GlobalMaxPool -> Dense -> Dropout -> Dense embedding.

    This version keeps that encoder shape and replaces cosine similarity against
    job-title embeddings with a single binary classification head.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 1000,
        kernel_size: int = 5,
        hidden_dim: int = 1000,
        projection_dim: int = 100,
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
        )
        self.hidden = nn.Linear(num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.classifier = nn.Linear(projection_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = torch.tanh(self.conv(embedded))
        pooled = F.max_pool1d(features, kernel_size=features.shape[-1]).squeeze(-1)
        hidden = torch.tanh(self.hidden(pooled))
        projected = F.relu(self.projection(self.dropout(hidden)))
        return self.classifier(projected).squeeze(-1)

