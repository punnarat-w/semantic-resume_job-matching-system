from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("deep_conv")
class DeepConvMatcher(ResumeRequirementModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 256,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # First convolution block
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)

        # Second convolution block (deeper)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters)

        # Multi‑kernel final layer
        self.convs = nn.ModuleList([
            nn.Conv1d(num_filters, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_filters, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)

        # First block
        out = F.relu(self.bn1(self.conv1(embedded)))
        out = F.relu(self.bn2(self.conv2(out)))

        # Multi‑kernel convolutions
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(out))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[-1]).squeeze(-1)
            conv_outs.append(pooled)

        features = torch.cat(conv_outs, dim=1)
        features = self.dropout(features)
        return self.classifier(features).squeeze(-1)