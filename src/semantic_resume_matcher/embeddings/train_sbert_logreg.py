from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from semantic_resume_matcher.data import RequirementExample, expand_records, load_requirement_records, split_records_by_resume
from semantic_resume_matcher.embeddings.features import build_pair_features
from semantic_resume_matcher.train import make_resume_snippet, write_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_sbert_logreg(
    train_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/sbert_logreg.yaml",
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

    encoder = load_sentence_transformer(
        model_name=config["embedding_model"]["name"],
        device=config["embedding_model"].get("device"),
    )
    train_features = encode_examples(encoder, train_examples, config)
    val_features = encode_examples(encoder, val_examples, config)

    classifier = LogisticRegression(**config["classifier"])
    classifier.fit(train_features, [example.label for example in train_examples])

    metrics, prediction_rows = evaluate(
        classifier=classifier,
        features=val_features,
        examples=val_examples,
        threshold=config["training"]["threshold"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "classifier.joblib")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "validation_predictions.jsonl", prediction_rows)
    write_jsonl(
        output_dir / "validation_failures.jsonl",
        [row for row in prediction_rows if not row["correct"]],
    )
    return metrics


def load_sentence_transformer(model_name: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for the SBERT embedding baseline. "
            "Install it with `pip install -r requirements.txt`."
        ) from error

    return SentenceTransformer(model_name, device=device)


def encode_examples(encoder, examples: list[RequirementExample], config: dict) -> np.ndarray:
    batch_size = config["embedding_model"]["batch_size"]
    normalize_embeddings = config["embedding_model"].get("normalize_embeddings", True)

    requirement_embeddings = encoder.encode(
        [example.requirement for example in examples],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    resume_embeddings = encoder.encode(
        [example.resume for example in examples],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return build_pair_features(requirement_embeddings, resume_embeddings)


def evaluate(
    classifier: LogisticRegression,
    features: np.ndarray,
    examples: list[RequirementExample],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    labels = [example.label for example in examples]
    probabilities = classifier.predict_proba(features)[:, 1].tolist()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sentence-BERT embeddings + Logistic Regression baseline.")
    parser.add_argument("--train-path", default="data/processed/resume_score_details.jsonl")
    parser.add_argument("--output-dir", default="artifacts/sbert_logreg")
    parser.add_argument("--config-path", default="configs/sbert_logreg.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_sbert_logreg(args.train_path, args.output_dir, args.config_path)
    print(json.dumps(metrics, indent=2))

