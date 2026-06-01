from __future__ import annotations

import numpy as np


def build_pair_features(requirement_embeddings: np.ndarray, resume_embeddings: np.ndarray) -> np.ndarray:
    absolute_difference = np.abs(requirement_embeddings - resume_embeddings)
    elementwise_product = requirement_embeddings * resume_embeddings
    cosine = cosine_similarity_column(requirement_embeddings, resume_embeddings)
    return np.concatenate(
        [
            requirement_embeddings,
            resume_embeddings,
            absolute_difference,
            elementwise_product,
            cosine,
        ],
        axis=1,
    )


def cosine_similarity_column(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1, keepdims=True)
    return numerator / np.clip(denominator, a_min=1e-12, a_max=None)

