from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semantic_resume_matcher.data import (
    RequirementDataset,
    RequirementExample,
    apply_evidence_retrieval,
    expand_records,
    iter_texts_for_vocab,
    load_requirement_records,
    split_records_by_resume,
)

from semantic_resume_matcher.models import ResumeRequirementModel, build_model
from semantic_resume_matcher.models import bow_mlp as _bow_mlp  # noqa: F401
from semantic_resume_matcher.models import ruozhengu_cnn as _ruozhengu_cnn  # noqa: F401
from semantic_resume_matcher.models import lstm_mlp as _lstm_mlp  # noqa: F401
from semantic_resume_matcher.models import text_cnn as _text_cnn  # noqa: F401
from semantic_resume_matcher.text import Vocabulary
from semantic_resume_matcher.models import attention_cnn as _attention_cnn  # noqa: F401
from semantic_resume_matcher.models import siamese_cnn as _siamese_cnn  # noqa: F401



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    train_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/baseline.yaml",
) -> dict[str, float]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    # Data preparation
    records = load_requirement_records(train_path)
    train_records, val_records = split_records_by_resume(
        records,
        validation_split=config["training"]["validation_split"],
        seed=config["seed"],
    )
    train_examples = expand_records(train_records)
    val_examples = expand_records(val_records)

    evidence_config = config.get("evidence", {})
    if evidence_config.get("enabled", False):
        top_k = int(evidence_config.get("top_k", 3))
        train_examples = apply_evidence_retrieval(train_examples, top_k=top_k)
        val_examples = apply_evidence_retrieval(val_examples, top_k=top_k)

    ## build the vocabulary from the training examples only, to avoid data leakage. The vocabulary will be used to convert tokens to ids for the model.
    vocab = Vocabulary.build(
        iter_texts_for_vocab(train_examples),
        min_freq=config["min_freq"],
        max_size=config.get("max_vocab_size"),
    )
    ## wrap into pytorch datasets and dataloaders for training and evaluation
    train_dataset = RequirementDataset(train_examples, vocab, max_length=config["max_length"])
    val_dataset = RequirementDataset(val_examples, vocab, max_length=config["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        name=config["model"]["name"],
        vocab_size=len(vocab.token_to_id),
        pad_id=vocab.pad_id,
        params=config["model"].get("params", {}),
    ).to(device)

    #TODO: consider using other optimizers or loss functions, and tuning hyperparameters such as learning rate, batch size, number of epochs, etc.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config["training"]["epochs"]):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=float(loss.detach().cpu()))

    metrics, prediction_rows = evaluate(
        model,
        val_loader,
        examples=val_examples,
        device=device,
        threshold=config["training"]["threshold"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "vocab.json").write_text(json.dumps(vocab.token_to_id, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "validation_predictions.jsonl", prediction_rows)
    write_jsonl(
        output_dir / "validation_failures.jsonl",
        [row for row in prediction_rows if not row["correct"]],
    )
    return metrics


train_baseline = train_model


def evaluate(
    model: ResumeRequirementModel,
    data_loader: DataLoader,
    examples: list[RequirementExample],
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"].to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["labels"].int().tolist())

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
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


def make_resume_snippet(resume: str, max_chars: int = 500) -> str:
    snippet = " ".join(resume.split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3] + "..."


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/raw/sample.jsonl")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--config-path", default="configs/baseline.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_model(args.train_path, args.output_dir, args.config_path)
    print(json.dumps(metrics, indent=2))
