from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from data.paths import RATINGS_PARQUET, TRAIN_PARQUET, TEST_PARQUET

def make_split(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    min_train: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = ratings.copy()
    df["_r"] = rng.random(len(df))
    df = df.sort_values(["user_idx", "_r"], kind="stable")
    grp = df.groupby("user_idx", sort=False)
    rank = grp.cumcount().to_numpy()
    size = grp["user_idx"].transform("size").to_numpy()

    n_test = np.ceil(size * test_frac).astype(int)
    n_test = np.minimum(n_test, np.maximum(size - min_train, 0))
    test_mask = rank < n_test
    cols = ["user_idx", "item_idx", "rating", "date"]
    test = df.loc[test_mask, cols].reset_index(drop=True)
    train = df.loc[~test_mask, cols].reset_index(drop=True)
    return train, test


def main() -> None:
    p = argparse.ArgumentParser(description="Create train/test parquet files.")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--min-train", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ratings = pd.read_parquet(RATINGS_PARQUET)
    train, test = make_split(ratings, args.test_frac, args.min_train, args.seed)
    train.to_parquet(TRAIN_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)
    print(f"train: {len(train):,} ratings | test: {len(test):,} ratings "
          f"({len(test) / len(ratings):.1%} held out)")
    print(f"  -> {TRAIN_PARQUET}\n  -> {TEST_PARQUET}")


if __name__ == "__main__":
    main()
