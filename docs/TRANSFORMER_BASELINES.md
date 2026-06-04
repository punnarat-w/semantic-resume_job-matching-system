# Transformer Cross-Encoder Baselines

A cross-encoder directly compares a minimum requirement and a resume in one transformer input:

```text
[CLS] requirement [SEP] resume [SEP]
```

The model outputs one binary logit for whether the resume satisfies that requirement.

This is a stronger fit for requirement-level matching than CNN/LSTM baselines because the transformer attention layers can compare tokens across the requirement and resume text.

## Train

```bash
python3 -m semantic_resume_matcher.transformer.train_cross_encoder \
  --train-path data/processed/resume_score_details.jsonl \
  --config-path configs/cross_encoder.yaml \
  --output-dir artifacts/cross_encoder
```

Artifacts:

- `model/`: fine-tuned Hugging Face model
- `tokenizer/`: tokenizer files
- `config.json`
- `metrics.json`
- `validation_predictions.jsonl`
- `validation_failures.jsonl`

## Inference

```python
from semantic_resume_matcher.transformer.inference import CrossEncoderMatcher

matcher = CrossEncoderMatcher.from_artifact_dir("artifacts/cross_encoder")
result = matcher.predict(
    minimum_requirements=["5+ years of Python experience"],
    resume="Resume text ..."
)
```

## Notes

The current implementation truncates long resumes to `max_length`. This is a clean first cross-encoder baseline. A stronger future version should split resumes into chunks, score each chunk against the requirement, then aggregate the top chunk scores.

