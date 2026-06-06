from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from data import paths

def load_ratings_file(filepath, chunkSize=4_000_000):
    all_chunks = []
    current_movie = None
    
    reader = pd.read_csv(
        filepath,
        header=None,
        names=["col1", "col2", "col3"],
        dtype="string",
        chunksize=chunkSize,
    )
    
    for chunk in reader:
        marker = chunk["col2"].isna()
        
        movie_ids = chunk["col1"].where(marker).str.rstrip(":")
        movie_ids = movie_ids.ffill()
        
        if current_movie is not None:
            movie_ids = movie_ids.fillna(current_movie)
        current_movie = movie_ids.iloc[-1]
        
        rating_rows = chunk[~marker]
        tmp_df = pd.DataFrame({
            "user_id": rating_rows["col1"].astype("int32"),
            "movie_id": movie_ids[~marker].astype("int32"),
            "rating": rating_rows["col2"].astype("int8"),
            "date": pd.to_datetime(rating_rows["col3"], format="%Y-%m-%d"),
        })
        all_chunks.append(tmp_df.reset_index(drop=True))
    
    return pd.concat(all_chunks, ignore_index=True)


def process_all_files(files_to_process):
    dfs = []
    for f in files_to_process:
        print(f"  working on {f.name} ...", flush=True)
        start = time.time()
        df = load_ratings_file(f)
        print(f"    got {len(df):,} ratings in {time.time() - start:.1f}s", flush=True)
        dfs.append(df)
    
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


def filter_ratings(df, min_user, min_movie, max_users=None, max_movies=None, iterations=2):
    for _ in range(iterations):
        movie_counts = df["movie_id"].value_counts()
        good_movies = movie_counts[movie_counts >= min_movie].index
        df = df[df["movie_id"].isin(good_movies)]
        
        user_counts = df["user_id"].value_counts()
        good_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(good_users)]
    
    if max_movies:
        top_m = df["movie_id"].value_counts().nlargest(max_movies).index
        df = df[df["movie_id"].isin(top_m)]
    if max_users:
        top_u = df["user_id"].value_counts().nlargest(max_users).index
        df = df[df["user_id"].isin(top_u)]
    
    return df.reset_index(drop=True)


def create_mappings(df):
    users = np.sort(df["user_id"].unique())
    movies = np.sort(df["movie_id"].unique())
    
    user_map = pd.Series(np.arange(len(users), dtype="int32"), index=users)
    movie_map = pd.Series(np.arange(len(movies), dtype="int32"), index=movies)
    
    remapped = pd.DataFrame({
        "user_idx": df["user_id"].map(user_map).astype("int32"),
        "item_idx": df["movie_id"].map(movie_map).astype("int32"),
        "rating": df["rating"].astype("int8"),
        "date": df["date"],
    })
    
    user_lookup = pd.DataFrame({
        "user_idx": np.arange(len(users), dtype="int32"),
        "user_id": users
    })
    item_lookup = pd.DataFrame({
        "item_idx": np.arange(len(movies), dtype="int32"),
        "movie_id": movies
    })
    
    return remapped, user_lookup, item_lookup


def get_movie_titles():
    titles = pd.read_csv(
        paths.MOVIE_TITLES,
        header=None,
        names=["movie_id", "year", "title"],
        encoding="latin-1",
        on_bad_lines="skip",
    )
    titles["movie_id"] = pd.to_numeric(titles["movie_id"], errors="coerce")
    titles = titles.dropna(subset=["movie_id"])
    titles["movie_id"] = titles["movie_id"].astype("int32")
    return titles


def build_dataset(
    file_list,
    min_user_ratings,
    min_movie_ratings,
    max_users=None,
    max_movies=None,
):
    paths.ensure_dirs()
    print("Parsing raw rating files ...", flush=True)
    df = process_all_files(file_list)
    raw_count = len(df)
    print(f"Parsed {raw_count:,} raw ratings, {df['user_id'].nunique():,} users, "
          f"{df['movie_id'].nunique():,} movies", flush=True)
    
    print("Filtering to active users / popular movies ...", flush=True)
    df = filter_ratings(df, min_user_ratings, min_movie_ratings, max_users, max_movies)
    kept_ratio = len(df) / raw_count
    print(f"Subset: {len(df):,} ratings ({kept_ratio:.1%} of raw), "
          f"{df['user_id'].nunique():,} users, {df['movie_id'].nunique():,} movies", flush=True)
    
    ratings, user_map, item_map = create_mappings(df)
    n_users = len(user_map)
    n_items = len(item_map)
    sparsity = 1 - len(ratings) / (n_users * n_items)
    
    titles = get_movie_titles()
    titles = titles[titles["movie_id"].isin(item_map["movie_id"])]
    titles = item_map.merge(titles, on="movie_id", how="left")
    
    ratings.to_parquet(paths.RATINGS_PARQUET, index=False)
    user_map.to_parquet(paths.USER_MAP_PARQUET, index=False)
    item_map.to_parquet(paths.ITEM_MAP_PARQUET, index=False)
    titles.to_parquet(paths.TITLES_PARQUET, index=False)
    
    meta = {
        "n_ratings": int(len(ratings)),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "sparsity": float(sparsity),
        "global_mean": float(ratings["rating"].mean()),
        "rating_min_date": str(ratings["date"].min().date()),
        "rating_max_date": str(ratings["date"].max().date()),
        "source_files": [f.name for f in file_list],
        "filters": {
            "min_user_ratings": min_user_ratings,
            "min_movie_ratings": min_movie_ratings,
            "max_users": max_users,
            "max_movies": max_movies,
        },
    }
    paths.META_JSON.write_text(json.dumps(meta, indent=2))
    print("\nSaved processed dataset:")
    print(f"  ratings : {len(ratings):,}")
    print(f"  users   : {n_users:,}")
    print(f"  items   : {n_items:,}")
    print(f"  sparsity: {sparsity:.4%}")
    print(f"  -> {paths.PROCESSED_DIR}")


files = [paths.COMBINED_FILES[i - 1] for i in [1]]
build_dataset(
    file_list=files,
    min_user_ratings=50,
    min_movie_ratings=2000,
    max_users=40000,
    max_movies=1500,
)