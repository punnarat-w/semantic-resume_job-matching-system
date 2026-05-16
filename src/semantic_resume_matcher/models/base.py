from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class ResumeRequirementModel(nn.Module):
    """Base class for models that score one requirement-resume pair."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

