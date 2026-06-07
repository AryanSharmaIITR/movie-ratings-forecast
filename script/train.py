from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import paths
from data.split import make_split
from evaluation.evaluate import evaluate_model
from models import MODELS

def load_data():
    if not paths.RATINGS_PARQUET.exists():
        print("Processed data not found. Run: python -m src.data.preprocess")
        exit(1)
    
    return pd.read_parquet(paths.RATINGS_PARQUET)


def build_model(name, quick_mode=False):
    model_class = MODELS[name]
    if name == "svd":
        return model_class(n_factors=50, n_epochs=8 if quick_mode else 25)
    if name == "als":
        return model_class(n_factors=40, n_iters=4 if quick_mode else 15)
    return model_class()

# ========== CONFIGURATION ==========
MODELS_TO_TRAIN = ["baseline", "als", "itemknn", "svd"]
TEST_FRAC = 0.2
K_VALUE = 10 
QUICK_MODE = False  

paths.ensure_dirs()
ratings =load_data()
meta =json.loads(paths.META_JSON.read_text())
n_users,n_items =meta["n_users"], meta["n_items"]
print(f"Dataset: {len(ratings):,} ratings | {n_users:,} users | {n_items:,} items")

print("\nSplitting data into train/test...")
train,test= make_split(ratings, TEST_FRAC)
train.to_parquet(paths.TRAIN_PARQUET, index=False)
test.to_parquet(paths.TEST_PARQUET, index=False)
print(f"Split: {len(train):,} train/{len(test):,} test\n")

results = []

for model_name in MODELS_TO_TRAIN:
    model = build_model(model_name, QUICK_MODE)
    print(f"=== {model.name} ===", flush=True)
    
    start_time = time.time()
    model.fit(train, n_users, n_items)
    fit_time = time.time() - start_time
    
    print("        evaluating...", flush=True)
    eval_results = evaluate_model(model, train, test, n_users, n_items, k=K_VALUE)
    eval_results["model"] = model.name
    eval_results["fit_seconds"] = round(fit_time, 1)
    results.append(eval_results)
    
    model_path = paths.MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    print(f"        fit={fit_time:.1f}s  RMSE={eval_results['RMSE']:.4f}  "
          f"MAP@{K_VALUE}={eval_results[f'MAP@{K_VALUE}']:.4f}  "
          f"NDCG@{K_VALUE}={eval_results[f'NDCG@{K_VALUE}']:.4f}\n", flush=True)
    
comparison_df = pd.DataFrame(results).set_index("model")
cols_to_keep = ["RMSE", "MAE", f"MAP@{K_VALUE}", f"NDCG@{K_VALUE}",
                f"Precision@{K_VALUE}", f"Recall@{K_VALUE}", f"HitRate@{K_VALUE}",
                f"Coverage@{K_VALUE}", "fit_seconds", "eval_seconds"]
cols_to_keep = [c for c in cols_to_keep if c in comparison_df.columns]
comparison_df = comparison_df[cols_to_keep]

comparison_df.to_csv(paths.RESULTS_DIR / "comparison.csv")
comparison_df.round(4).to_json(paths.RESULTS_DIR / "comparison.json", orient="index", indent=2)