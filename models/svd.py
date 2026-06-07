from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseRecommender

class FunkSVD(BaseRecommender):
    name = "Funk-SVD (MF + SGD)"
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs:int = 25,
        lr: float =0.05,
        reg: float= 0.05,
        batch_size:int = 16_384,
    ):
        self.n_factors = n_factors
        self.n_epochs =n_epochs
        self.lr = lr
        self.reg =reg
        self.batch_size = batch_size

    def fit(self,train_df: pd.DataFrame, n_users:int,n_items: int) -> "FunkSVD":
        self.n_users= n_users
        self.n_items= n_items
        rng =np.random.default_rng(42)
        u =train_df["user_idx"].to_numpy()
        i =train_df["item_idx"].to_numpy()
        r= train_df["rating"].to_numpy(dtype=np.float32)
        n =len(r)

        self.mu =float(r.mean())
        self.b_u =np.zeros(n_users, dtype=np.float32)
        self.b_i=np.zeros(n_items, dtype=np.float32)
        scale = 0.1/np.sqrt(self.n_factors)
        self.P =rng.normal(0,scale, (n_users,self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0,scale, (n_items,self.n_factors)).astype(np.float32)
        lr, reg = self.lr,self.reg
        f=self.n_factors
        
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                ub, ib, rb = u[idx], i[idx], r[idx]

                pu,qi = self.P[ub], self.Q[ib]
                pred= self.mu+self.b_u[ub]+self.b_i[ib]+np.sum(pu * qi, axis=1)
                err = (rb-pred).astype(np.float32)

                uu, inv_u= np.unique(ub,return_inverse=True)
                ui, inv_i= np.unique(ib,return_inverse=True)
                cu =np.bincount(inv_u).astype(np.float32)
                ci =np.bincount(inv_i).astype(np.float32)

                gb_u= np.zeros(len(uu), dtype=np.float32)
                gb_i= np.zeros(len(ui), dtype=np.float32)
                gP = np.zeros((len(uu),f), dtype=np.float32)
                gQ = np.zeros((len(ui),f), dtype=np.float32)

                np.add.at(gb_u, inv_u, err)
                np.add.at(gb_i,inv_i, err)
                np.add.at(gP, inv_u, err[:, None] *qi)
                np.add.at(gQ, inv_i,err[:,None] * pu)

                self.b_u[uu]+=lr*(gb_u/cu-reg* self.b_u[uu])
                self.b_i[ui]+=lr*(gb_i/ci-reg * self.b_i[ui])
                self.P[uu]+= lr * (gP/cu[:, None] -reg* self.P[uu])
                self.Q[ui] += lr* (gQ/ ci[:,None] - reg* self.Q[ui])

            tr = self._train_rmse(u, i, r)
            print(f"    [svd] epoch {epoch + 1:>2}/{self.n_epochs}  train RMSE={tr:.5f}", flush=True)

        return self

    def _train_rmse(self, u,i,r) -> float:
        pred = self.predict_pairs(u,i)
        return float(np.sqrt(np.mean((r -pred) **2)))

    def predict_pairs(self, users: np.ndarray,items:np.ndarray) ->np.ndarray:
        pred = (
            self.mu
            + self.b_u[users]
            + self.b_i[items]
            + np.sum(self.P[users] * self.Q[items], axis=1)
        )

        return self._clip(pred)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        pred = (
            self.mu
            + self.b_u[users][:, None]
            + self.b_i[None, :]
            + self.P[users] @ self.Q.T
        )

        return pred
