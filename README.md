# Semantic Resume-Job Matching System

Binary requirement-level matching for job minimum requirements against a candidate resume.

Given a job's `minimum_requirements` and a resume, the system predicts whether the resume meets each requirement and extracts basic personal information.

## Example

```json
{
  "input": {
    "minimum_requirements": [
      "5+ years of software engineering experience",
      "Strong understanding of IT infrastructure"
    ],
    "resume": "Muhammad Talha Riaz ... talhariaz9969@gmail.com ..."
  },
  "output": {
    "scores": {
      "requirements": [
        {
          "criteria": "5+ years of software engineering experience",
          "meets": false,
          "probability": 0.42
        }
      ]
    },
    "personal_info": {
      "name": "Muhammad Talha Riaz",
      "email": "talhariaz9969@gmail.com"
    }
  }
}
```

## Repo Structure

```text
.
├── configs/
│   ├── baseline.yaml
│   ├── bow_mlp.yaml
│   ├── lstm_mlp.yaml
│   └── reference_cnn.yaml
│   └── tfidf_logreg.yaml
├── data/
│   ├── processed/
│   └── raw/
│       └── sample.jsonl
├── notebooks/
│   ├── 01_training_baseline.ipynb
│   ├── 01_training_classical_baseline.ipynb
│   └── 02_inference_baseline.ipynb
├── src/
│   └── semantic_resume_matcher/
│       ├── data.py
│       ├── inference.py
│       ├── models/
│       ├── personal_info.py
│       ├── text.py
│       └── train.py
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For notebooks:

```bash
python3 -m ipykernel install --user --name semantic-resume-matcher
jupyter lab
```

## Data Format

Use JSONL where each line is one job-resume example:

```json
{"minimum_requirements":["5+ years of Python experience","AWS experience"],"resume":"Resume text ...","labels":[1,0]}
```

`labels[i]` must correspond to `minimum_requirements[i]`.

The loader also supports the Hugging Face `netsol/resume-score-details` JSON shape:

```json
{
  "input": {
    "minimum_requirements": ["5+ years of experience..."],
    "resume": "Resume text..."
  },
  "output": {
    "scores": {
      "requirements": [{"criteria": "5+ years of experience...", "meets": false}]
    }
  }
}
```

You can pass either a single `.json`, a `.jsonl`, or a directory containing JSON/JSONL files.

The raw Hugging Face folder includes invalid/incomplete samples. Prepare a clean training file first:

```bash
python3 scripts/prepare_hf_dataset.py \
  --input-dir data/raw/resume-score-details \
  --output-path data/processed/resume_score_details.jsonl
```

## Train

Use `notebooks/01_training_baseline.ipynb`, or run a neural model:

```bash
python3 -m semantic_resume_matcher.train \
  --train-path data/processed/resume_score_details.jsonl \
  --config-path configs/baseline.yaml \
  --output-dir artifacts/baseline
```

Train the classical TF-IDF + Logistic Regression baseline:

```bash
python3 -m semantic_resume_matcher.classical.train_tfidf_logreg \
  --train-path data/processed/resume_score_details.jsonl \
  --config-path configs/tfidf_logreg.yaml \
  --output-dir artifacts/tfidf_logreg
```

Training writes:

- `model.pt`: learned model weights
- `vocab.json`: word vocabulary
- `config.json`: resolved model/training config
- `metrics.json`: validation metrics
- `validation_predictions.jsonl`: every validation requirement prediction
- `validation_failures.jsonl`: only the misclassified validation requirements

Each prediction row includes the requirement, true label, predicted label, probability, and a resume snippet so you can inspect failure cases.

The train/validation split is grouped by normalized resume text before expanding into requirement-level examples. This prevents the same resume from appearing in both training and validation, even when one resume appears in multiple job records.

## Inference

Use `notebooks/02_inference_baseline.ipynb`, or:

```python
from semantic_resume_matcher.inference import ResumeJobMatcher

matcher = ResumeJobMatcher.from_artifact_dir("artifacts/baseline")
result = matcher.predict(
    minimum_requirements=["5+ years of Python experience"],
    resume="Resume text ..."
)
```

## Baseline

Models operate on a paired input:

```text
[CLS] minimum requirement [SEP] resume text
```

Each model performs one binary classification per requirement.

Available model configs:

- `configs/baseline.yaml`: multi-kernel TextCNN with kernel sizes `[3, 4, 5]`.
- `configs/ruozhengu_cnn.yaml`: adapted version of the CNN repo architecture with a binary classification head.
- `configs/lstm_mlp.yaml`: adapted version of the Kanhaiya Jee LSTM architecture with a binary classification head.
- `configs/bow_mlp.yaml`: simple bag-of-words MLP example for testing the plug-in path.

The adapted RuoZhengu CNN uses:

```text
Embedding
-> Conv1D(filters=1000, kernel_size=5, activation=tanh)
-> GlobalMaxPool
-> Dense(1000, activation=tanh)
-> Dropout(0.3)
-> Dense(100, activation=relu)
-> Dense(1) binary logit
```

The adapted Kanhaiya Jee LSTM uses:

```text
Embedding(vocab_size=10000, embedding_dim=128)
-> LSTM(hidden_dim=128)
-> Dense(64, activation=relu)
-> Dense(1) binary logit
```

The original LSTM used `Dense(3, activation=softmax)` for a 3-class task. This repo replaces that final layer with `Dense(1)` and trains with binary cross-entropy because each minimum requirement is a true/false decision.


## Adding Models

Model architectures live in `src/semantic_resume_matcher/models/` and are selected by YAML:

```yaml
model:
  name: text_cnn
  params:
    embedding_dim: 128
    num_filters: 128
```

See `docs/ADDING_MODELS.md` for the plug-in pattern.
Classical ML baselines use a separate scikit-learn pipeline. See `docs/CLASSICAL_BASELINES.md`.