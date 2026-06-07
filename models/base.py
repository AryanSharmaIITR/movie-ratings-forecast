from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse

RATING_MIN, RATING_MAX = 1.0, 5.0

def build_csr(train_df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            train_df["rating"].to_numpy(dtype=np.float32),
            (train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy()),
        ),
        shape=(n_users,n_items),
    )

def topk_from_scores(
    scores: np.ndarray,
    seen_csr: sparse.csr_matrix,
    user_indices: np.ndarray,
    k: int,
) -> dict[int, np.ndarray]:
    scores = scores.copy()
    for row, u in enumerate(user_indices):
        seen = seen_csr.indices[seen_csr.indptr[u]:seen_csr.indptr[u + 1]]
        scores[row, seen] = -np.inf

    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    out: dict[int, np.ndarray] = {}
    
    for row,u in enumerate(user_indices):
        cols = part[row]
        order = np.argsort(-scores[row,cols])
        out[int(u)] = cols[order]
    
    return out


class BaseRecommender:
    name = "base"
    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> "BaseRecommender":
        raise NotImplementedError

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_users(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _clip(x: np.ndarray) -> np.ndarray:
        return np.clip(x, RATING_MIN, RATING_MAX)
