# Classical ML Baselines

Classical models reuse the same requirement-level data loader, but they do not use the PyTorch model registry.

The training split is still grouped by normalized resume text before expanding to requirement-level examples, so the same resume does not appear in both train and validation.

Use them for baselines such as:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Count vectors + Naive Bayes

## TF-IDF + Logistic Regression

This baseline formats each sample as:

```text
Requirement: <minimum requirement>
Resume: <resume text>
```

Then it trains:

```text
TfidfVectorizer(ngram_range=(1, 2))
-> LogisticRegression(class_weight="balanced")
```

Train:
Run on `notebooks/01_training_classical_baseline.ipynb` or

```bash
python3 -m semantic_resume_matcher.classical.train_tfidf_logreg \
  --train-path data/processed/resume_score_details.jsonl \
  --config-path configs/tfidf_logreg.yaml \
  --output-dir artifacts/tfidf_logreg
```

Artifacts:

- `model.joblib`
- `config.json`
- `metrics.json`
- `validation_predictions.jsonl`
- `validation_failures.jsonl`

## Why Separate From `ADDING_MODELS.md`?

`docs/ADDING_MODELS.md` is for PyTorch models that accept token IDs and output one binary logit.

Classical ML models use scikit-learn vectorizers and estimators, so they have a separate training/inference path while sharing the same dataset format and evaluation exports.
