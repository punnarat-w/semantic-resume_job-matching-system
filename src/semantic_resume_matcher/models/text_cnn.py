from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("text_cnn")
class TextCNNMatcher(ResumeRequirementModel):
    """TextCNN baseline for binary requirement-resume matching."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            features = F.relu(conv(embedded))
            pooled = F.max_pool1d(features, kernel_size=features.shape[-1]).squeeze(-1)
            pooled_outputs.append(pooled)
        features = torch.cat(pooled_outputs, dim=1)
        return self.classifier(self.dropout(features)).squeeze(-1)

