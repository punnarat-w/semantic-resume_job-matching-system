from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("siamese_cnn")
class SiameseCNNMatcher(ResumeRequirementModel):
    """Dual-encoder CNN for requirement-resume matching.

    The input is still the repo's normal paired format:
        <cls> requirement <sep> resume

    This model splits the sequence at <sep>, encodes the requirement and resume
    separately with a shared CNN encoder, then classifies their match.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 128,
        dropout: float = 0.3,
        sep_id: int = 3,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.sep_id = sep_id
        self.kernel_sizes = list(kernel_sizes)
        self.min_sequence_length = max(self.kernel_sizes)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in self.kernel_sizes
            ]
        )

        encoder_dim = num_filters * len(self.kernel_sizes)
        pair_dim = encoder_dim * 4

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        requirement_ids, resume_ids = self._split_pair_batch(input_ids)

        requirement_vec = self._encode_text(requirement_ids)
        resume_vec = self._encode_text(resume_ids)

        features = torch.cat(
            [
                requirement_vec,
                resume_vec,
                torch.abs(requirement_vec - resume_vec),
                requirement_vec * resume_vec,
            ],
            dim=1,
        )

        return self.classifier(features).squeeze(-1)

    def _encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []
        for conv in self.convs:
            convolved = F.relu(conv(embedded))
            pooled = F.max_pool1d(convolved, kernel_size=convolved.size(2)).squeeze(2)
            pooled_outputs.append(pooled)

        return torch.cat(pooled_outputs, dim=1)

    def _split_pair_batch(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        requirement_sequences = []
        resume_sequences = []

        for row in input_ids:
            sep_positions = (row == self.sep_id).nonzero(as_tuple=False)

            if len(sep_positions) == 0:
                split_index = row.size(0) // 2
            else:
                split_index = int(sep_positions[0].item())

            requirement = row[1:split_index]
            resume = row[split_index + 1 :]
            requirement = requirement[requirement != self.pad_id]
            resume = resume[resume != self.pad_id]
            requirement_sequences.append(self._ensure_min_length(requirement, row.device))
            resume_sequences.append(self._ensure_min_length(resume, row.device))

        requirement_batch = self._pad_sequences(requirement_sequences)
        resume_batch = self._pad_sequences(resume_sequences)
        return requirement_batch, resume_batch

    def _ensure_min_length(self, sequence: torch.Tensor, device: torch.device) -> torch.Tensor:
        if sequence.numel() == 0:
            sequence = torch.tensor([self.pad_id], dtype=torch.long, device=device)

        if sequence.numel() >= self.min_sequence_length:
            return sequence

        padding = torch.full(
            (self.min_sequence_length - sequence.numel(),),
            self.pad_id,
            dtype=torch.long,
            device=device,
        )
        return torch.cat([sequence, padding], dim=0)

    def _pad_sequences(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        max_length = max(sequence.numel() for sequence in sequences)
        padded_sequences = []

        for sequence in sequences:
            if sequence.numel() < max_length:
                padding = torch.full(
                    (max_length - sequence.numel(),),
                    self.pad_id,
                    dtype=torch.long,
                    device=sequence.device,
                )
                sequence = torch.cat([sequence, padding], dim=0)
            padded_sequences.append(sequence)
        return torch.stack(padded_sequences, dim=0)