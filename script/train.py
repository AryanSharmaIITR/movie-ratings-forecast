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
    return model_class()

# ========== CONFIGURATION - EDIT THESE ==========
MODELS_TO_TRAIN = ["baseline"] 
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