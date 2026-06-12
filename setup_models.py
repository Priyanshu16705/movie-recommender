"""
setup_models.py
===============
Run this script ONCE if the models/ folder is empty (e.g. after a fresh clone
or on Streamlit Cloud where large files aren't committed).

It:
  1. Generates a synthetic MovieLens-style dataset if CSVs are missing.
  2. Runs the full training pipeline.

Usage:
    python setup_models.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")


def models_exist() -> bool:
    required = ["tfidf.pkl", "sim_matrix.npy", "svd_model.pkl",
                "movie_df.pkl", "metrics.json"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def data_exists() -> bool:
    return all(
        os.path.exists(os.path.join(DATA_DIR, f))
        for f in ["movies.csv", "ratings.csv", "tags.csv"]
    )


def generate_synthetic_data():
    """Generate small synthetic MovieLens-style CSVs."""
    import pandas as pd
    import numpy as np
    import random

    print("Generating synthetic dataset …")
    random.seed(42)
    np.random.seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)

    genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                   'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
                   'Sci-Fi', 'Thriller']

    movies = [
        {'movieId': i,
         'title': f'Movie {i} ({random.randint(1990, 2023)})',
         'genres': '|'.join(random.sample(genres_list, random.randint(1, 3)))}
        for i in range(1, 501)
    ]

    ratings = [
        {'userId': u,
         'movieId': random.randint(1, 500),
         'rating': random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
         'timestamp': random.randint(1_000_000_000, 1_700_000_000)}
        for u in range(1, 1001)
        for _ in range(random.randint(20, 60))
    ]

    tags = [
        {'userId': random.randint(1, 1000),
         'movieId': random.randint(1, 500),
         'tag': random.choice(['fun', 'classic', 'slow', 'intense',
                               'funny', 'scary', 'romantic', 'suspenseful']),
         'timestamp': random.randint(1_000_000_000, 1_700_000_000)}
        for _ in range(3000)
    ]

    pd.DataFrame(movies).to_csv(os.path.join(DATA_DIR, 'movies.csv'), index=False)
    pd.DataFrame(ratings).to_csv(os.path.join(DATA_DIR, 'ratings.csv'), index=False)
    pd.DataFrame(tags).to_csv(os.path.join(DATA_DIR, 'tags.csv'), index=False)
    print("  ✅  Synthetic data saved to data/")


if __name__ == "__main__":
    if models_exist():
        print("✅  Models already exist — nothing to do.")
        sys.exit(0)

    if not data_exists():
        generate_synthetic_data()

    # Run training
    import train
    train.train()
