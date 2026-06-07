from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseRecommender

class BaselineBias(BaseRecommender):
    name = "Baseline (bias)"
    def __init__(self, reg_user: float = 10.0, reg_item: float = 25.0):
        self.reg_user = reg_user
        self.reg_item = reg_item

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> "BaselineBias":
        self.n_users = n_users
        self.n_items = n_items
        u = train_df["user_idx"].to_numpy()
        i = train_df["item_idx"].to_numpy()
        r = train_df["rating"].to_numpy(dtype=np.float64)

        self.mu = float(r.mean())

        dev = r - self.mu
        item_sum = np.bincount(i, weights=dev, minlength=n_items)
        item_cnt = np.bincount(i, minlength=n_items)
        self.b_i = item_sum / (self.reg_item + item_cnt)

        dev_u = r - self.mu - self.b_i[i]
        user_sum = np.bincount(u, weights=dev_u, minlength=n_users)
        user_cnt = np.bincount(u, minlength=n_users)
        self.b_u = user_sum / (self.reg_user + user_cnt)
    
        return self 

    def predict_pairs_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.mu + self.b_u[users] + self.b_i[items]

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self._clip(self.predict_pairs_raw(users, items))

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.mu + self.b_u[users][:, None] + self.b_i[None, :]
