from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from .base import BaseRecommender
from .baseline import BaselineBias

class ALS(BaseRecommender):
    name = "ALS (MF)"
    def __init__(
        self,
        n_factors: int = 40,
        n_iters: int = 15,
        reg: float = 50.0,
        reg_user: float = 10.0,
        reg_item: float = 25.0,
    ):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.reg_user = reg_user
        self.reg_item = reg_item

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> "ALS":
        
        self.n_users = n_users
        self.n_items = n_items
        rng = np.random.default_rng(42)

        self.baseline = BaselineBias(self.reg_user, self.reg_item).fit(train_df, n_users, n_items)
        u = train_df["user_idx"].to_numpy()
        i = train_df["item_idx"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float32)
        resid = (r - self.baseline.predict_pairs_raw(u, i)).astype(np.float32)

        Ru = sparse.csr_matrix((resid, (u, i)), shape=(n_users, n_items))
        Ri = Ru.T.tocsr()

        self.P = (rng.normal(0, 0.1, (n_users, self.n_factors)) / np.sqrt(self.n_factors)).astype(np.float32)
        self.Q = (rng.normal(0, 0.1, (n_items, self.n_factors)) / np.sqrt(self.n_factors)).astype(np.float32)
        eye = self.reg * np.eye(self.n_factors, dtype=np.float64)

        for it in range(self.n_iters):

            self._als_step(Ru, self.Q, self.P, eye)   
            self._als_step(Ri, self.P, self.Q, eye)  

            if self.verbose:
                pred = self.predict_pairs(u, i)
                rmse = float(np.sqrt(np.mean((r - pred) ** 2)))

                print(f"    [als] iter {it + 1:>2}/{self.n_iters}  train RMSE={rmse:.4f}", flush=True)
        return self

    @staticmethod
    def _als_step(R: sparse.spmatrix, fixed: np.ndarray, target: np.ndarray, eye: np.ndarray) -> None:
        
        Rcsr = R.tocsr()
        indptr, indices, data = Rcsr.indptr, Rcsr.indices, Rcsr.data
        for m in range(R.shape[0]):
            lo, hi = indptr[m], indptr[m + 1]
            if lo == hi:
                continue
            cols = indices[lo:hi]
            s = data[lo:hi]
            Y = fixed[cols].astype(np.float64)           
            A = Y.T @ Y + eye                            
            b = Y.T @ s                                  
            target[m] = np.linalg.solve(A, b).astype(np.float32)

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        base = self.baseline.predict_pairs_raw(users, items)
        inter = np.sum(self.P[users] * self.Q[items], axis=1)
        return self._clip(base + inter)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        base = self.baseline.mu + self.baseline.b_u[users][:, None] + self.baseline.b_i[None, :]
        return base + self.P[users] @ self.Q.T
