
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender, build_csr
from .baseline import BaselineBias


class ItemKNN(BaseRecommender):
    name = "Item-based KNN CF"
    def __init__(self, k_neighbors: int = 200, reg_user: float = 10.0, reg_item: float = 25.0):
        self.k_neighbors = k_neighbors
        self.reg_user = reg_user
        self.reg_item = reg_item

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> "ItemKNN":

        self.n_users = n_users
        self.n_items = n_items

        self.baseline = BaselineBias(self.reg_user, self.reg_item).fit(train_df, n_users, n_items)

        R = build_csr(train_df, n_users, n_items)
        self.R = R
        R_coo = R.tocoo()

        item_mean = np.asarray(R.sum(axis=0)).ravel() / np.maximum(R.getnnz(axis=0), 1)

        centred = sparse.csr_matrix(
            (R_coo.data - item_mean[R_coo.col], (R_coo.row, R_coo.col)),
            shape=R.shape,
        )

        b = self.baseline
        resid = R_coo.data - (b.mu + b.b_u[R_coo.row] + b.b_i[R_coo.col])
        self.R_resid = sparse.csr_matrix((resid.astype(np.float32), (R_coo.row, R_coo.col)),
                                         shape=R.shape)
        self.R_indicator = (R != 0).astype(np.float32)

        sim = cosine_similarity(centred.T.tocsr(), dense_output=True)
        np.fill_diagonal(sim, 0.0)
        self.sim = self._prune_topn(sim, self.k_neighbors)

        self._scores: np.ndarray | None = None
        return self

    @staticmethod
    def _prune_topn(sim: np.ndarray, n: int) -> sparse.csr_matrix:
        n = min(n, sim.shape[1] - 1)
        keep_idx = np.argpartition(-np.abs(sim), kth=n, axis=1)[:, :n]
        rows = np.repeat(np.arange(sim.shape[0]), n)
        cols = keep_idx.ravel()
        vals = sim[rows, cols]

        return sparse.csr_matrix((vals, (rows, cols)), shape=sim.shape)

    def _full_scores(self) -> np.ndarray:
        if self._scores is None:
            numer = (self.R_resid @ self.sim.T).toarray()
            denom = (self.R_indicator @ np.abs(self.sim).T).toarray()
            base = (
                self.baseline.mu
                + self.baseline.b_u[:, None]
                + self.baseline.b_i[None, :]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                neigh = np.where(denom > 0, numer / denom, 0.0)
            self._scores = self._clip(base + neigh).astype(np.float32)

        return self._scores

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_scores"] = None
        
        return state

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self._full_scores()[users, items]

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self._full_scores()[users]
