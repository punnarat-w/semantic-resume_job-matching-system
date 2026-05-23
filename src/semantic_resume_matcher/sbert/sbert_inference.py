from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
from semantic_resume_matcher.personal_info import extract_personal_info

class SBERTJobMatcher:
    def __init__(self, model_path, threshold=0.5):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.clf = joblib.load(model_path)
        self.threshold = threshold

    def predict(self, minimum_requirements, resume):
        req_emb = self.encoder.encode(minimum_requirements)
        res_emb = self.encoder.encode([resume] * len(minimum_requirements))
        diff = np.abs(req_emb - res_emb)
        prod = req_emb * res_emb
        X = np.concatenate([req_emb, res_emb, diff, prod], axis=1)
        probabilities = self.clf.predict_proba(X)[:, 1]
        rows = []
        for req, prob in zip(minimum_requirements, probabilities):
            rows.append({
                "criteria": req,
                "meets": bool(prob >= self.threshold),
                "probability": round(prob, 4)
            })
        return {
            "scores": {"requirements": rows},
            "personal_info": extract_personal_info(resume)
        }