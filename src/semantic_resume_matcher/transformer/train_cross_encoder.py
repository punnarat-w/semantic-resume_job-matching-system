from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from semantic_resume_matcher.data import RequirementExample, expand_records, load_requirement_records, split_records_by_resume
from semantic_resume_matcher.train import make_resume_snippet, write_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CrossEncoderDataset(Dataset):
    def __init__(self, examples: list[RequirementExample], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.requirement,
            example.resume,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(example.label, dtype=torch.float32)
        return item


def train_cross_encoder(
    train_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/cross_encoder.yaml",
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

    tokenizer, model = load_transformer(config)
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_dataset = CrossEncoderDataset(train_examples, tokenizer, max_length=config["model"]["max_length"])
    val_dataset = CrossEncoderDataset(val_examples, tokenizer, max_length=config["model"]["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    scheduler = build_linear_warmup_scheduler(
        optimizer=optimizer,
        total_steps=len(train_loader) * config["training"]["epochs"],
        warmup_ratio=config["training"].get("warmup_ratio", 0.0),
    )
    criterion = build_loss(train_examples, config, device)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch in progress:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=float(loss.detach().cpu()))

    metrics, prediction_rows = evaluate(
        model=model,
        data_loader=val_loader,
        examples=val_examples,
        device=device,
        threshold=config["training"]["threshold"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "validation_predictions.jsonl", prediction_rows)
    write_jsonl(
        output_dir / "validation_failures.jsonl",
        [row for row in prediction_rows if not row["correct"]],
    )
    return metrics


def load_transformer(config: dict):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required for the cross-encoder baseline. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from error

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


def build_loss(train_examples: list[RequirementExample], config: dict, device: torch.device) -> nn.Module:
    if not config["training"].get("use_pos_weight", True):
        return nn.BCEWithLogitsLoss()

    labels = [example.label for example in train_examples]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return nn.BCEWithLogitsLoss()

    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(remaining_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model,
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
            batch_labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.int().tolist())

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
    parser = argparse.ArgumentParser(description="Train a transformer cross-encoder baseline.")
    parser.add_argument("--train-path", default="data/processed/resume_score_details.jsonl")
    parser.add_argument("--output-dir", default="artifacts/cross_encoder")
    parser.add_argument("--config-path", default="configs/cross_encoder.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_cross_encoder(args.train_path, args.output_dir, args.config_path)
    print(json.dumps(metrics, indent=2))
