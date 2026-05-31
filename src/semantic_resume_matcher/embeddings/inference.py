from __future__ import annotations

import json
from pathlib import Path

import joblib

from semantic_resume_matcher.embeddings.features import build_pair_features
from semantic_resume_matcher.embeddings.train_sbert_logreg import load_sentence_transformer
from semantic_resume_matcher.personal_info import extract_personal_info


class SbertLogRegMatcher:
    def __init__(self, encoder, classifier, config: dict) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.config = config
        self.threshold = config["training"]["threshold"]

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "SbertLogRegMatcher":
        artifact_dir = Path(artifact_dir)
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        classifier = joblib.load(artifact_dir / "classifier.joblib")
        encoder = load_sentence_transformer(
            model_name=config["embedding_model"]["name"],
            device=config["embedding_model"].get("device"),
        )
        return cls(encoder=encoder, classifier=classifier, config=config)

    def predict(self, minimum_requirements: list[str], resume: str) -> dict:
        batch_size = self.config["embedding_model"]["batch_size"]
        normalize_embeddings = self.config["embedding_model"].get("normalize_embeddings", True)

        requirement_embeddings = self.encoder.encode(
            minimum_requirements,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        resume_embeddings = self.encoder.encode(
            [resume] * len(minimum_requirements),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        features = build_pair_features(requirement_embeddings, resume_embeddings)
        probabilities = self.classifier.predict_proba(features)[:, 1].tolist()
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

