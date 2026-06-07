from __future__ import annotations
import numpy as np
RELEVANCE_THRESHOLD = 3.5

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true -y_pred) **2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true= np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true- y_pred)))

def _dcg(relevances: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances * discounts))


def average_precision(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec= recommended[:k]
    hits = 0
    score =0.0
    for i, item in enumerate(rec):
        if item in relevant:
            hits+= 1
            score += hits /(i+1)
    denom = min(len(relevant),k)

    return score/denom if denom else 0.0


def ndcg(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec= recommended[:k]
    gains= np.array([1.0 if item in relevant else 0.0 for item in rec])
    ideal = np.ones(min(len(relevant), k))
    idcg =_dcg(ideal)

    return _dcg(gains)/ idcg if idcg else 0.0


def precision_recall(recommended: np.ndarray, relevant: set[int], k: int) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    rec = recommended[:k]
    hits = sum(1 for item in rec if item in relevant)
    precision = hits / k
    recall = hits/ len(relevant)

    return precision,recall

def evaluate_ranking(
    topk_per_user: dict[int, np.ndarray],
    relevant_per_user: dict[int, set[int]],
    k: int,
    n_items: int,
) -> dict[str,float]:
    aps, ndcgs, precs, recs, hits = [], [], [], [], []
    recommended_items: set[int] = set()

    for user,relevant in relevant_per_user.items():
        if not relevant:
            continue
        rec = topk_per_user.get(user)
        if rec is None:
            continue
        recommended_items.update(rec[:k].tolist())
        aps.append(average_precision(rec, relevant, k))
        ndcgs.append(ndcg(rec, relevant,k))
        p, r = precision_recall(rec,relevant, k)
        precs.append(p)
        recs.append(r)
        hits.append(1.0 if p>0 else 0.0)

    return {
        f"MAP@{k}": float(np.mean(aps)) if aps else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"Recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"Coverage@{k}": len(recommended_items) / n_items,
        "n_eval_users": len(aps),
    }
