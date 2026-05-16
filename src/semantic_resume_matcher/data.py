from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from semantic_resume_matcher.text import Vocabulary


@dataclass(frozen=True)
class RequirementExample:
    requirement: str
    resume: str
    label: int


def load_requirement_examples(path: str | Path) -> list[RequirementExample]:
    path = Path(path)
    if path.is_dir():
        examples: list[RequirementExample] = []
        for json_path in sorted(path.glob("*.json")):
            examples.extend(_load_json_file(json_path))
        for jsonl_path in sorted(path.glob("*.jsonl")):
            examples.extend(_load_jsonl_file(jsonl_path))
        return examples

    if path.suffix == ".json":
        return _load_json_file(path)
    return _load_jsonl_file(path)


def _load_jsonl_file(path: Path) -> list[RequirementExample]:
    examples: list[RequirementExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            examples.extend(_record_to_examples(record, source=f"{path}:{line_number}"))
    return examples


def _load_json_file(path: Path) -> list[RequirementExample]:
    record = json.loads(path.read_text(encoding="utf-8"))
    return _record_to_examples(record, source=str(path))


def _record_to_examples(record: dict, source: str) -> list[RequirementExample]:
    if "input" in record and "output" in record:
        input_record = record["input"]
        requirements = input_record.get("minimum_requirements", [])
        resume = input_record.get("resume", "")
        requirement_scores = record.get("output", {}).get("scores", {}).get("requirements", [])
        labels_by_criteria = {item.get("criteria"): int(bool(item.get("meets"))) for item in requirement_scores}
        labels = [labels_by_criteria.get(requirement) for requirement in requirements]
    else:
        requirements = record["minimum_requirements"]
        resume = record["resume"]
        labels = record["labels"]

    if len(requirements) != len(labels):
        raise ValueError(f"{source}: requirements and labels must have same length.")
    if any(label is None for label in labels):
        raise ValueError(f"{source}: every minimum requirement must have a matching output score.")

    return [
        RequirementExample(requirement=requirement, resume=resume, label=int(label))
        for requirement, label in zip(requirements, labels, strict=True)
    ]


def iter_texts_for_vocab(examples: list[RequirementExample]) -> Iterator[str]:
    for example in examples:
        yield example.requirement
        yield example.resume


class RequirementDataset(Dataset):
    def __init__(self, examples: list[RequirementExample], vocab: Vocabulary, max_length: int) -> None:
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        input_ids = self.vocab.encode_pair(example.requirement, example.resume, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(example.label, dtype=torch.float32),
        }
