"""
data_preprocessing.py
=====================
Handles all data loading, cleaning, merging, and exploratory data analysis
for the Movie Recommendation System.

Steps covered:
  1. Load movies.csv, ratings.csv, tags.csv from disk.
  2. Handle missing values (drop or fill).
  3. Merge datasets into a single enriched DataFrame.
  4. Build a 'content' feature column (genres + tags) for TF-IDF.
  5. Provide basic EDA summary statistics.
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
class DataLoader:
    """Loads and validates the three MovieLens CSV files."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Generic CSV loader with basic validation."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{filename} not found in {self.data_dir}. "
                "Run the data download step first."
            )
        df = pd.read_csv(path)
        print(f"  Loaded {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    def load_movies(self) -> pd.DataFrame:
        return self._load_csv("movies.csv")

    def load_ratings(self) -> pd.DataFrame:
        return self._load_csv("ratings.csv")

    def load_tags(self) -> pd.DataFrame:
        return self._load_csv("tags.csv")


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------
class DataPreprocessor:
    """
    Cleans individual DataFrames and merges them into one enriched table.

    The merged DataFrame has columns:
        movieId, title, genres, content, avg_rating, num_ratings
    """

    # ---- individual cleaning -----------------------------------------------

    @staticmethod
    def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
        """
        - Drop rows with null movieId or title.
        - Fill missing genres with '(no genres listed)'.
        - Strip whitespace from string columns.
        """
        df = df.copy()
        before = len(df)
        df.dropna(subset=["movieId", "title"], inplace=True)
        df["genres"] = df["genres"].fillna("(no genres listed)").str.strip()
        df["title"] = df["title"].str.strip()
        df["movieId"] = df["movieId"].astype(int)
        print(f"  clean_movies: {before} → {len(df)} rows (dropped {before - len(df)} nulls)")
        return df.reset_index(drop=True)

    @staticmethod
    def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
        """
        - Drop rows with null userId / movieId / rating.
        - Clip ratings to [0.5, 5.0].
        - Cast types.
        """
        df = df.copy()
        before = len(df)
        df.dropna(subset=["userId", "movieId", "rating"], inplace=True)
        df["rating"] = df["rating"].clip(0.5, 5.0)
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        print(f"  clean_ratings: {before} → {len(df)} rows (dropped {before - len(df)} nulls)")
        return df.reset_index(drop=True)

    @staticmethod
    def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
        """
        - Drop rows with null tag.
        - Lower-case and strip tags.
        """
        df = df.copy()
        before = len(df)
        df.dropna(subset=["tag"], inplace=True)
        df["tag"] = df["tag"].str.lower().str.strip()
        print(f"  clean_tags: {before} → {len(df)} rows (dropped {before - len(df)} nulls)")
        return df.reset_index(drop=True)

    # ---- aggregation helpers -----------------------------------------------

    @staticmethod
    def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
        """Concatenate all tags for each movie into a single string."""
        return (
            tags_df.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x))
            .reset_index()
            .rename(columns={"tag": "all_tags"})
        )

    @staticmethod
    def aggregate_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Compute average rating and rating count per movie."""
        return (
            ratings_df.groupby("movieId")["rating"]
            .agg(avg_rating="mean", num_ratings="count")
            .reset_index()
        )

    # ---- full pipeline ------------------------------------------------------

    def build_movie_df(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge movies + aggregated tags + aggregated ratings into one DataFrame.
        Also creates a 'content' column = genres text + tags text, used by TF-IDF.
        """
        movies = self.clean_movies(movies_df)
        ratings = self.clean_ratings(ratings_df)
        tags = self.clean_tags(tags_df)

        tag_agg = self.aggregate_tags(tags)
        rating_agg = self.aggregate_ratings(ratings)

        # Merge tags
        df = movies.merge(tag_agg, on="movieId", how="left")
        df["all_tags"] = df["all_tags"].fillna("")

        # Merge rating stats
        df = df.merge(rating_agg, on="movieId", how="left")
        df["avg_rating"] = df["avg_rating"].fillna(0.0).round(2)
        df["num_ratings"] = df["num_ratings"].fillna(0).astype(int)

        # Build content feature: replace '|' in genres with spaces so each
        # genre token is treated independently by TF-IDF.
        df["genre_text"] = df["genres"].str.replace("|", " ", regex=False).str.lower()
        df["content"] = df["genre_text"] + " " + df["all_tags"]
        df["content"] = df["content"].str.strip()

        print(f"\n  Final movie DataFrame: {df.shape[0]} movies × {df.shape[1]} columns")
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------------------
class EDA:
    """Provides quick summary statistics for loaded DataFrames."""

    @staticmethod
    def summarize(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
        """Return a dict of key EDA metrics."""
        stats = {
            "total_movies": len(movies_df),
            "total_ratings": len(ratings_df),
            "total_users": ratings_df["userId"].nunique(),
            "avg_rating_global": round(ratings_df["rating"].mean(), 3),
            "rating_std": round(ratings_df["rating"].std(), 3),
            "ratings_per_user_mean": round(
                ratings_df.groupby("userId")["rating"].count().mean(), 1
            ),
            "ratings_per_movie_mean": round(
                ratings_df.groupby("movieId")["rating"].count().mean(), 1
            ),
            "sparsity_pct": round(
                100
                * (
                    1
                    - len(ratings_df)
                    / (ratings_df["userId"].nunique() * movies_df["movieId"].nunique())
                ),
                2,
            ),
        }
        return stats

    @staticmethod
    def genre_distribution(movies_df: pd.DataFrame) -> pd.Series:
        """Count movies per genre (a movie can appear in multiple genre buckets)."""
        genre_series = movies_df["genres"].str.split("|").explode()
        return genre_series.value_counts()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def load_and_prepare(data_dir: str = DATA_DIR):
    """
    One-call helper used by train.py and app.py.
    Returns (movie_df, ratings_df, movies_raw).
    """
    print("=== Loading data ===")
    loader = DataLoader(data_dir)
    movies_raw = loader.load_movies()
    ratings_raw = loader.load_ratings()
    tags_raw = loader.load_tags()

    print("\n=== Preprocessing ===")
    preprocessor = DataPreprocessor()
    movie_df = preprocessor.build_movie_df(movies_raw, ratings_raw, tags_raw)

    print("\n=== EDA Summary ===")
    stats = EDA.summarize(movies_raw, ratings_raw)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return movie_df, ratings_raw, movies_raw
