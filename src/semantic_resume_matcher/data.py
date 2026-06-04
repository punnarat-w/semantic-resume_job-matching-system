from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from semantic_resume_matcher.text import Vocabulary

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RequirementExample:
    requirement: str
    resume: str
    label: int


@dataclass(frozen=True)
class RequirementRecord:
    requirements: list[str]
    resume: str
    labels: list[int]


def load_requirement_examples(path: str | Path) -> list[RequirementExample]:
    return expand_records(load_requirement_records(path))


def load_requirement_records(path: str | Path) -> list[RequirementRecord]:
    path = Path(path)
    if path.is_dir():
        records: list[RequirementRecord] = []
        for json_path in sorted(path.glob("*.json")):
            records.extend(_load_json_file(json_path))
        for jsonl_path in sorted(path.glob("*.jsonl")):
            records.extend(_load_jsonl_file(jsonl_path))
        return records

    if path.suffix == ".json":
        return _load_json_file(path)
    return _load_jsonl_file(path)


def _load_jsonl_file(path: Path) -> list[RequirementRecord]:
    records: list[RequirementRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(_record_to_requirement_record(record, source=f"{path}:{line_number}"))
    return records


def _load_json_file(path: Path) -> list[RequirementRecord]:
    record = json.loads(path.read_text(encoding="utf-8"))
    return [_record_to_requirement_record(record, source=str(path))]


def _record_to_requirement_record(record: dict, source: str) -> RequirementRecord:
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

    return RequirementRecord(
        requirements=list(requirements),
        resume=resume,
        labels=[int(label) for label in labels],
    )


def expand_records(records: list[RequirementRecord]) -> list[RequirementExample]:
    return [
        RequirementExample(requirement=requirement, resume=record.resume, label=label)
        for record in records
        for requirement, label in zip(record.requirements, record.labels, strict=True)
    ]

def apply_evidence_retrieval(
    examples: list[RequirementExample],
    top_k: int = 3,
) -> list[RequirementExample]:
    """Replace each full resume with the top-k most relevant resume sentences.

    Relevance is computed using TF-IDF cosine similarity between the requirement
    and each sentence in the resume.
    """
    return [
        RequirementExample(
            requirement=example.requirement,
            resume=get_top_k_evidence(example.requirement, example.resume, top_k=top_k),
            label=example.label,
        )
        for example in examples
    ]


def get_top_k_evidence(requirement: str, resume: str, top_k: int = 3) -> str:
    sentences = split_resume_sentences(resume)

    if top_k <= 0:
        return resume

    if len(sentences) <= top_k:
        return resume

    try:
        texts = [requirement] + sentences
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts)
        requirement_vec = tfidf[0]
        sentence_vecs = tfidf[1:]
        similarities = cosine_similarity(requirement_vec, sentence_vecs).flatten()
        top_indices = similarities.argsort()[-top_k:]
        top_indices = sorted(top_indices)
        selected_sentences = [sentences[index] for index in top_indices]
        return " ".join(selected_sentences)

    except ValueError:
        return resume


def split_resume_sentences(resume: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+|;+", resume)
    sentences = []
    
    for sentence in raw_sentences:
        cleaned = " ".join(sentence.split())
        if len(cleaned) >= 20:
            sentences.append(cleaned)

    if sentences:
        return sentences

    cleaned_resume = " ".join(resume.split())
    return [cleaned_resume] if cleaned_resume else []

def split_records_by_resume(
    records: list[RequirementRecord],
    validation_split: float,
    seed: int,
) -> tuple[list[RequirementRecord], list[RequirementRecord]]:
    groups = [normalize_text(record.resume) for record in records]
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=seed)
    train_indices, val_indices = next(splitter.split(records, groups=groups))
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]
    return train_records, val_records


def normalize_text(text: str) -> str:
    return " ".join(text.split())


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
