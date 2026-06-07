from __future__ import annotations
import time
import numpy as np
import pandas as pd
from evaluation import metrics
from models.base import build_csr, topk_from_scores

def _relevant_per_user(test_df: pd.DataFrame) -> dict[int, set[int]]:
    rel = test_df[test_df["rating"]>= metrics.RELEVANCE_THRESHOLD]
    return {
        int(u):set(grp.to_numpy().tolist())
        for u, grp in rel.groupby("user_idx")["item_idx"]
    }

def evaluate_model(
    model,
    train_df: pd.DataFrame,
    test_df:pd.DataFrame,
    n_users: int,
    n_items:int,
    k: int= 10,
    batch_size: int =4000,
) -> dict[str, float]:
    results: dict[str,float] = {}
    t0 = time.time()
    tu = test_df["user_idx"].to_numpy()
    ti = test_df["item_idx"].to_numpy()
    tr = test_df["rating"].to_numpy(dtype=np.float64)
    pred = model.predict_pairs(tu, ti)
    results["RMSE"] = metrics.rmse(tr, pred)
    results["MAE"] = metrics.mae(tr, pred)

    seen = build_csr(train_df, n_users, n_items)
    relevant = _relevant_per_user(test_df)
    eval_users = np.array(sorted(relevant.keys()))
    topk: dict[int, np.ndarray] = {}

    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start:start + batch_size]
        scores = model.score_users(batch)
        topk.update(topk_from_scores(scores, seen, batch, k))

    ranking = metrics.evaluate_ranking(topk, relevant, k, n_items)
    results.update(ranking)
    results["eval_seconds"] = round(time.time() - t0, 1)

    return results
