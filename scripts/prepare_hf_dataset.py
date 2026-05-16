from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_hf_dataset(input_dir: str | Path, output_path: str | Path) -> dict[str, int]:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "files_seen": 0,
        "records_written": 0,
        "requirements_written": 0,
        "skipped_invalid": 0,
        "skipped_malformed": 0,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        for path in sorted(input_dir.glob("*.json")):
            stats["files_seen"] += 1
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                prepared = prepare_record(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                stats["skipped_malformed"] += 1
                continue

            if prepared is None:
                stats["skipped_invalid"] += 1
                continue

            output_file.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            stats["records_written"] += 1
            stats["requirements_written"] += len(prepared["minimum_requirements"])

    return stats


def prepare_record(record: dict) -> dict | None:
    output = record["output"]
    if output.get("valid_resume_and_jd") is False:
        return None

    input_record = record["input"]
    requirements = input_record["minimum_requirements"]
    resume = input_record["resume"]
    requirement_scores = output["scores"]["requirements"]

    labels_by_criteria = {
        score["criteria"]: int(bool(score["meets"]))
        for score in requirement_scores
        if "criteria" in score and "meets" in score
    }
    labels = [labels_by_criteria.get(requirement) for requirement in requirements]

    if not requirements or not resume or any(label is None for label in labels):
        return None

    return {
        "minimum_requirements": requirements,
        "resume": resume,
        "labels": labels,
        "source_valid_resume_and_jd": output.get("valid_resume_and_jd"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare netsol/resume-score-details JSON files for training.")
    parser.add_argument("--input-dir", default="data/raw/resume-score-details")
    parser.add_argument("--output-path", default="data/processed/resume_score_details.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = prepare_hf_dataset(args.input_dir, args.output_path)
    print(json.dumps(stats, indent=2))

