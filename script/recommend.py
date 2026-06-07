from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from data import paths
from models.base import build_csr, topk_from_scores


def load_artifacts(model_name):
    with open(paths.MODELS_DIR / f"{model_name}.pkl", "rb") as fh:
        model = pickle.load(fh)
    titles = pd.read_parquet(paths.TITLES_PARQUET).set_index("item_idx")
    user_map = pd.read_parquet(paths.USER_MAP_PARQUET)
    ratings = pd.read_parquet(paths.RATINGS_PARQUET)
    
    return model,titles,user_map,ratings

def get_movie_title(titles, item_idx):
    if item_idx in titles.index:
        row= titles.loc[item_idx]
        year = "" if pd.isna(row.get("year")) else f" ({int(row['year'])})"
        return f"{row['title']}{year}"
    
    return f"item#{item_idx}"


def recommend_for_user(user_idx, model, titles, ratings, n_items, k=10):
    n_users =int(ratings["user_idx"].max()) + 1
    seen =build_csr(ratings, n_users, n_items)
    scores =model.score_users(np.array([user_idx]))
    topk =topk_from_scores(scores, seen, np.array([user_idx]), k)[user_idx]
    preds =model.predict_pairs(np.full(k, user_idx), topk)
    
    recommendations =[(int(i), get_movie_title(titles,int(i)),float(s)) 
                      for i,s in zip(topk,preds)]
    liked = (ratings[ratings["user_idx"]==user_idx]
             .sort_values("rating",ascending=False)
             .head(k))
    
    liked_movies = [(get_movie_title(titles,int(r.item_idx)), int(r.rating)) 
                    for r in liked.itertuples()]
    
    return recommendations,liked_movies

USER_ID = 42  # internal user_idx (0..n_users-1)
USE_RAW_ID = False  # set to True if USER_ID is a raw Netflix customer id
MODEL_NAME = "svd"  # options: "baseline", "itemknn", "svd", "als"
K_RECOMMENDATIONS = 10


print(f"\nLoading model '{MODEL_NAME}' and data...")
model, titles, user_map, ratings = load_artifacts(MODEL_NAME)
n_items = int(ratings["item_idx"].max()) + 1

user_idx = USER_ID
if USE_RAW_ID:
    match = user_map[user_map["user_id"] == USER_ID]
    if match.empty:
        print(f"Raw user id {USER_ID} is not in the processed subset.")
        exit(1)
    user_idx = int(match["user_idx"].iloc[0])
    print(f"Raw id {USER_ID} -> internal idx {user_idx}")

recommendations, liked_movies = recommend_for_user(user_idx, model, titles, ratings, n_items, K_RECOMMENDATIONS)

print(f"\nUser idx {user_idx} — model: {model.name}")
print("\nBecause you rated highly:")
for movie_title, rating in liked_movies:
    print(f"  {rating}★  {movie_title}")

print(f"\nTop-{K_RECOMMENDATIONS} recommendations:")
for rank, (idx, movie_title, score) in enumerate(recommendations, 1):
    print(f"  {rank:>2}. {movie_title}   (predicted {score:.2f}★)")