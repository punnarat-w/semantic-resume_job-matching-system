from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("attention_cnn")
class AttentionCNNMatcher(ResumeRequirementModel):
    """TextCNN with global attention over kernel outputs.

    This model extends the multi‑kernel CNN by adding a self‑attention layer
    that learns to weight each time step's combined filter outputs before pooling.
    """

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
        total_filters = num_filters * len(kernel_sizes)
        self.attn_linear = nn.Linear(total_filters, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_filters, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(embedded))
            conv_outputs.append(out)

        max_len = max(out.shape[-1] for out in conv_outputs)
        padded = []
        for out in conv_outputs:
            pad_size = max_len - out.shape[-1]
            if pad_size > 0:
                out = F.pad(out, (0, pad_size))
            padded.append(out)
        combined = torch.cat(padded, dim=1)

        attn_scores = self.attn_linear(combined.permute(0, 2, 1))
        attn_weights = F.softmax(attn_scores, dim=1)
        attended = torch.sum(combined.permute(0, 2, 1) * attn_weights, dim=1)

        features = self.dropout(attended)
        return self.classifier(features).squeeze(-1)