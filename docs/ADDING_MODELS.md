# Adding Model Architectures

The repo uses a small model registry so teammates can add architectures without changing the training or inference code.

Existing registered models:

- `text_cnn`: multi-kernel TextCNN baseline.
- `ruozhengu_cnn`: adapted version of the CNN repo architecture, ending in a binary classifier.
- `lstm_mlp`: adapted version of the Kanhaiya Jee LSTM architecture, ending in a binary classifier.
- `bow_mlp`: simple example architecture.

## 1. Create a Model File

Create a file under `src/semantic_resume_matcher/models/`.

```python
from __future__ import annotations

import torch
from torch import nn

from semantic_resume_matcher.models.base import ResumeRequirementModel
from semantic_resume_matcher.models.registry import register_model


@register_model("my_model")
class MyModel(ResumeRequirementModel):
    def __init__(self, vocab_size: int, pad_id: int = 0, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(input_ids).mean(dim=1)
        return self.classifier(pooled).squeeze(-1)
```

The model must accept:

- `vocab_size`
- `pad_id`
- any extra parameters from the YAML config

The model must return one logit per requirement-resume pair.

## 2. Import the Model Module

Registering happens when the module is imported. Add an import in both `train.py` and `inference.py`:

```python
from semantic_resume_matcher.models import my_model as _my_model  # noqa: F401
```

## 3. Add a Config

Create a config such as `configs/my_model.yaml`:

```yaml
seed: 42
max_length: 512
min_freq: 1

model:
  name: my_model
  params:
    hidden_dim: 128

training:
  batch_size: 16
  epochs: 5
  learning_rate: 0.001
  validation_split: 0.2
  threshold: 0.5
```

## 4. Train

```bash
python3 -m semantic_resume_matcher.train \
  --train-path data/raw/sample.jsonl \
  --config-path configs/my_model.yaml \
  --output-dir artifacts/my_model
```
