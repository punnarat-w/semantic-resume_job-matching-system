from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from semantic_resume_matcher.data import RequirementExample, expand_records, load_requirement_records, split_records_by_resume
from semantic_resume_matcher.train import make_resume_snippet, write_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_tfidf_logreg(
    train_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/tfidf_logreg.yaml",
) -> dict[str, float]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(config["seed"])

    records = load_requirement_records(train_path)
    train_records, val_records = split_records_by_resume(
        records,
        validation_split=config["training"]["validation_split"],
        seed=config["seed"],
    )
    train_examples = expand_records(train_records)
    val_examples = expand_records(val_records)


    vectorizer_config = dict(config["vectorizer"])
    if "ngram_range" in vectorizer_config:
        vectorizer_config["ngram_range"] = tuple(vectorizer_config["ngram_range"])

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**vectorizer_config)),
            ("classifier", LogisticRegression(**config["classifier"])),
        ]
    )
    pipeline.fit(
        [format_pair(example) for example in train_examples],
        [example.label for example in train_examples],
    )

    metrics, prediction_rows = evaluate(
        pipeline=pipeline,
        examples=val_examples,
        threshold=config["training"]["threshold"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "validation_predictions.jsonl", prediction_rows)
    write_jsonl(
        output_dir / "validation_failures.jsonl",
        [row for row in prediction_rows if not row["correct"]],
    )
    return metrics


def evaluate(
    pipeline: Pipeline,
    examples: list[RequirementExample],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    labels = [example.label for example in examples]
    texts = [format_pair(example) for example in examples]
    probabilities = pipeline.predict_proba(texts)[:, 1].tolist()
    predictions = [int(probability >= threshold) for probability in probabilities]

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "num_validation_examples": len(labels),
        "num_failures": sum(int(label != prediction) for label, prediction in zip(labels, predictions, strict=True)),
    }
    rows = [
        {
            "index": index,
            "requirement": example.requirement,
            "label": label,
            "prediction": prediction,
            "probability": round(probability, 6),
            "correct": label == prediction,
            "resume_snippet": make_resume_snippet(example.resume),
        }
        for index, (example, label, prediction, probability) in enumerate(
            zip(examples, labels, predictions, probabilities, strict=True)
        )
    ]
    return metrics, rows


def format_pair(example: RequirementExample) -> str:
    return f"Requirement: {example.requirement}\nResume: {example.resume}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--train-path", default="data/processed/resume_score_details.jsonl")
    parser.add_argument("--output-dir", default="artifacts/tfidf_logreg")
    parser.add_argument("--config-path", default="configs/tfidf_logreg.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_tfidf_logreg(args.train_path, args.output_dir, args.config_path)
    print(json.dumps(metrics, indent=2))
