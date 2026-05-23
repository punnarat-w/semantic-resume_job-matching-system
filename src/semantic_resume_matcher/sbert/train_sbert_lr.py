import json
import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from semantic_resume_matcher.data import load_requirement_records, split_records_by_resume, expand_records

def train_sbert_lr(train_path, output_dir, config):
    # Load and split exactly like train.py
    records = load_requirement_records(train_path)
    train_records, val_records = split_records_by_resume(
        records,
        validation_split=config["validation_split"],
        seed=config["seed"],
    )
    train_examples = expand_records(train_records)
    val_examples = expand_records(val_records)

    # Extract texts and labels
    train_requirements = [ex.requirement for ex in train_examples]
    train_resumes = [ex.resume for ex in train_examples]
    train_labels = [ex.label for ex in train_examples]

    val_requirements = [ex.requirement for ex in val_examples]
    val_resumes = [ex.resume for ex in val_examples]
    val_labels = [ex.label for ex in val_examples]

    # Encode with SBERT
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Encoding training set...")
    train_req_emb = encoder.encode(train_requirements, show_progress_bar=True)
    train_res_emb = encoder.encode(train_resumes, show_progress_bar=True)
    print("Encoding validation set...")
    val_req_emb = encoder.encode(val_requirements, show_progress_bar=True)
    val_res_emb = encoder.encode(val_resumes, show_progress_bar=True)

    # Build features: [req, res, diff, product]
    def build_features(req_emb, res_emb):
        diff = np.abs(req_emb - res_emb)
        prod = req_emb * res_emb
        return np.concatenate([req_emb, res_emb, diff, prod], axis=1)

    X_train = build_features(train_req_emb, train_res_emb)
    X_val = build_features(val_req_emb, val_res_emb)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # Train logistic regression (balanced class weights)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # Predict and compute probabilities
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "num_validation_examples": len(y_val),
        "num_failures": int((y_val != y_pred).sum()),
    }
    print(f"Validation accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # Save model and metadata
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "sbert_lr_model.pkl")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save validation failures with probabilities
    with open(output_dir / "validation_failures.jsonl", "w") as f:
        for ex, label, pred, prob in zip(val_examples, y_val, y_pred, y_pred_proba):
            if label != pred:
                row = {
                    "requirement": ex.requirement,
                    "label": int(label),
                    "prediction": int(pred),
                    "probability": float(prob),
                    "resume_snippet": ex.resume[:300],
                }
                f.write(json.dumps(row) + "\n")
    return metrics

if __name__ == "__main__":
    config = {
        "seed": 42,
        "validation_split": 0.2,
    }
    train_sbert_lr(
        train_path="data/processed/resume_score_details.jsonl",
        output_dir="artifacts/sbert_lr",
        config=config
    )