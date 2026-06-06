from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "DataSet"
COMBINED_FILES = [DATASET_DIR / f"combined_data_{i}.txt" for i in range(1, 5)]
MOVIE_TITLES = DATASET_DIR / "movie_titles.csv"

PROCESSED_DIR = ROOT / "DataSet" / "processed"
MODELS_DIR = ROOT / "results" / "Artifacts"
RESULTS_DIR = ROOT / "results"
RATINGS_PARQUET = PROCESSED_DIR / "ratings.parquet"
USER_MAP_PARQUET = PROCESSED_DIR / "user_map.parquet"
ITEM_MAP_PARQUET = PROCESSED_DIR / "item_map.parquet"
TITLES_PARQUET = PROCESSED_DIR / "titles.parquet"
META_JSON = PROCESSED_DIR / "meta.json"

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
