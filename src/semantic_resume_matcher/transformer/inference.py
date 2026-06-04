from __future__ import annotations

import json
from pathlib import Path

import torch

from semantic_resume_matcher.personal_info import extract_personal_info


class CrossEncoderMatcher:
    def __init__(self, tokenizer, model, config: dict, device: torch.device | None = None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.threshold = config["training"]["threshold"]
        self.device = device or torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "CrossEncoderMatcher":
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "transformers is required for cross-encoder inference. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from error

        artifact_dir = Path(artifact_dir)
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir / "model")
        return cls(tokenizer=tokenizer, model=model, config=config)

    def predict(self, minimum_requirements: list[str], resume: str) -> dict:
        probabilities: list[float] = []
        batch_size = self.config["training"]["batch_size"]
        for start in range(0, len(minimum_requirements), batch_size):
            batch_requirements = minimum_requirements[start : start + batch_size]
            encoded = self.tokenizer(
                batch_requirements,
                [resume] * len(batch_requirements),
                truncation=True,
                padding=True,
                max_length=self.config["model"]["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits.squeeze(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())

        rows = [
            {
                "criteria": requirement,
                "meets": probability >= self.threshold,
                "probability": round(probability, 4),
            }
            for requirement, probability in zip(minimum_requirements, probabilities, strict=True)
        ]
        return {
            "scores": {"requirements": rows},
            "personal_info": extract_personal_info(resume),
        }

