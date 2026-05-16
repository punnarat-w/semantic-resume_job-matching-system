from __future__ import annotations

from pathlib import Path

import joblib

from semantic_resume_matcher.personal_info import extract_personal_info


class TfidfLogRegMatcher:
    def __init__(self, pipeline, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "TfidfLogRegMatcher":
        artifact_dir = Path(artifact_dir)
        pipeline = joblib.load(artifact_dir / "model.joblib")
        return cls(pipeline=pipeline)

    def predict(self, minimum_requirements: list[str], resume: str) -> dict:
        texts = [format_pair(requirement, resume) for requirement in minimum_requirements]
        probabilities = self.pipeline.predict_proba(texts)[:, 1].tolist()
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


def format_pair(requirement: str, resume: str) -> str:
    return f"Requirement: {requirement}\nResume: {resume}"

