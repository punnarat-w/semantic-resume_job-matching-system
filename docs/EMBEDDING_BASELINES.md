# Embedding Baselines

Embedding baselines use a pretrained sentence embedding model to represent the requirement and resume, then train a lightweight classifier.

## Sentence-BERT + Logistic Regression

This baseline uses:

```text
requirement -> SentenceTransformer -> requirement embedding
resume      -> SentenceTransformer -> resume embedding
```

Then it builds pair features:

```text
[requirement_embedding, resume_embedding, abs_difference, elementwise_product, cosine_similarity]
```

Finally it trains logistic regression for the binary `meets` label.

Train:

```bash
python3 -m semantic_resume_matcher.embeddings.train_sbert_logreg \
  --train-path data/processed/resume_score_details.jsonl \
  --config-path configs/sbert_logreg.yaml \
  --output-dir artifacts/sbert_logreg
```

Inference:

```python
from semantic_resume_matcher.embeddings.inference import SbertLogRegMatcher

matcher = SbertLogRegMatcher.from_artifact_dir("artifacts/sbert_logreg")
result = matcher.predict(
    minimum_requirements=["5+ years of Python experience"],
    resume="Resume text ..."
)
```

Artifacts:

- `classifier.joblib`
- `config.json`
- `metrics.json`
- `validation_predictions.jsonl`
- `validation_failures.jsonl`

The split is grouped by normalized resume text, matching the neural and classical baselines.

