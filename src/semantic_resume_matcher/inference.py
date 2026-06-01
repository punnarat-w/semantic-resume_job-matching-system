from __future__ import annotations

import json
from pathlib import Path

import torch

from semantic_resume_matcher.models import ResumeRequirementModel, build_model
from semantic_resume_matcher.models import bow_mlp as _bow_mlp  # noqa: F401
from semantic_resume_matcher.models import ruozhengu_cnn as _ruozhengu_cnn  # noqa: F401
from semantic_resume_matcher.models import lstm_mlp as _lstm_mlp  # noqa: F401
from semantic_resume_matcher.models import text_cnn as _text_cnn  # noqa: F401
from semantic_resume_matcher.models import attention_cnn as _attention_cnn
from semantic_resume_matcher.personal_info import extract_personal_info
from semantic_resume_matcher.text import Vocabulary
from semantic_resume_matcher.models import siamese_cnn as _siamese_cnn  # noqa: F401


class ResumeJobMatcher:
    def __init__(
        self,
        model: ResumeRequirementModel,
        vocab: Vocabulary,
        max_length: int,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.max_length = max_length
        self.threshold = threshold

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "ResumeJobMatcher":
        artifact_dir = Path(artifact_dir)
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        token_to_id = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
        vocab = Vocabulary(token_to_id=token_to_id)
        model = build_model(
            name=config["model"]["name"],
            vocab_size=len(vocab.token_to_id),
            pad_id=vocab.pad_id,
            params=config["model"].get("params", {}),
        )
        model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
        return cls(
            model=model,
            vocab=vocab,
            max_length=config["max_length"],
            threshold=config["training"]["threshold"],
        )

    def predict(self, minimum_requirements: list[str], resume: str) -> dict:
        rows = []
        for requirement in minimum_requirements:
            input_ids = self.vocab.encode_pair(requirement, resume, self.max_length)
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                probability = float(torch.sigmoid(self.model(tensor))[0].cpu())
            rows.append(
                {
                    "criteria": requirement,
                    "meets": probability >= self.threshold,
                    "probability": round(probability, 4),
                }
            )

        return {
            "scores": {"requirements": rows},
            "personal_info": extract_personal_info(resume),
        }
