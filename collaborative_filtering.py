"""
collaborative_filtering.py
==========================
Collaborative Filtering via Matrix Factorization (SVD) using the Surprise library.

How it works:
  1. Load the user-item rating matrix from ratings_df.
  2. Tune and train an SVD model on the full dataset.
  3. Predict the rating a given user would give to any unseen movie.
  4. Rank all unseen movies by predicted rating → top-K recommendations.
  5. Handle cold-start users (no rating history) gracefully.

Evaluation:
  - 5-fold cross-validation reporting RMSE and MAE.
  - Precision@K and Recall@K computed on a held-out test split.
"""

import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RATING_SCALE = (0.5, 5.0)


class CollaborativeFilteringRecommender:
    """
    SVD-based collaborative filtering.

    Attributes
    ----------
    model       : trained Surprise SVD model
    trainset    : Surprise Trainset (full data, used for predictions)
    all_movie_ids : complete set of movieIds in the dataset
    """

    def __init__(self):
        self.model: SVD | None = None
        self.trainset = None
        self.all_movie_ids: set = set()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def fit(self, ratings_df: pd.DataFrame):
        """
        Train SVD on the full ratings DataFrame.

        Parameters
        ----------
        ratings_df : DataFrame with columns [userId, movieId, rating]
        """
        print("  [CollabFilter] Preparing Surprise dataset …")
        reader = Reader(rating_scale=RATING_SCALE)

        # Surprise expects columns in order: user, item, rating
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader
        )

        # Build the full trainset (used for production predictions)
        self.trainset = data.build_full_trainset()
        self.all_movie_ids = set(ratings_df["movieId"].unique())

        print("  [CollabFilter] Training SVD model …")
        self.model = SVD(
            n_factors=100,      # latent factors
            n_epochs=20,        # SGD iterations
            lr_all=0.005,       # learning rate
            reg_all=0.02,       # regularisation
            random_state=42,
            verbose=False,
        )
        self.model.fit(self.trainset)
        print("  [CollabFilter] Training complete.")
        return self

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, models_dir: str = MODELS_DIR):
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(models_dir, "svd_model.pkl"))
        joblib.dump(self.trainset, os.path.join(models_dir, "svd_trainset.pkl"))
        print(f"  [CollabFilter] Models saved to {models_dir}")

    def load(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(os.path.join(models_dir, "svd_model.pkl"))
        self.trainset = joblib.load(os.path.join(models_dir, "svd_trainset.pkl"))
        # Reconstruct all_movie_ids from the trainset
        self.all_movie_ids = {
            self.trainset.to_raw_iid(iid)
            for iid in self.trainset.all_items()
        }
        print(f"  [CollabFilter] Models loaded from {models_dir}")
        return self

    # -------------------------------------------------------------------------
    # Prediction & Recommendation
    # -------------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating user_id would give to movie_id.
        Returns the global mean if either is unknown (cold-start fallback).
        """
        if self.model is None:
            raise RuntimeError("Call fit() or load() first.")
        pred = self.model.predict(str(user_id), str(movie_id))
        return round(pred.est, 3)

    def recommend(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        movie_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend top_k unseen movies for a given user, ranked by
        predicted SVD rating.

        Parameters
        ----------
        user_id    : target user
        ratings_df : full ratings DataFrame (to identify already-seen movies)
        movie_df   : enriched movie DataFrame (for title, genres)
        top_k      : number of recommendations

        Returns
        -------
        DataFrame [movieId, title, genres, predicted_rating]
        """
        if self.model is None:
            raise RuntimeError("Call fit() or load() first.")

        # Movies the user has already rated
        seen = set(
            ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].tolist()
        )
        unseen = self.all_movie_ids - seen

        if not unseen:
            # All movies seen – just return top-rated ones
            unseen = self.all_movie_ids

        # Predict for every unseen movie
        predictions = [
            (mid, self.predict_rating(user_id, mid)) for mid in unseen
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:top_k]

        top_ids = [mid for mid, _ in top]
        top_ratings = {mid: r for mid, r in top}

        result = movie_df[movie_df["movieId"].isin(top_ids)][
            ["movieId", "title", "genres", "avg_rating"]
        ].copy()
        result["predicted_rating"] = result["movieId"].map(top_ratings)
        result = result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
        return result

    def is_cold_start(self, user_id: int, ratings_df: pd.DataFrame) -> bool:
        """Return True if the user has fewer than 5 ratings (cold-start)."""
        count = (ratings_df["userId"] == user_id).sum()
        return count < 5

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate_cv(self, ratings_df: pd.DataFrame) -> dict:
        """
        Run 5-fold cross-validation and return mean RMSE and MAE.
        """
        print("  [CollabFilter] Running 5-fold cross-validation …")
        reader = Reader(rating_scale=RATING_SCALE)
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader
        )
        results = cross_validate(
            SVD(n_factors=100, n_epochs=20, random_state=42),
            data,
            measures=["RMSE", "MAE"],
            cv=5,
            verbose=False,
        )
        metrics = {
            "RMSE": round(float(np.mean(results["test_rmse"])), 4),
            "MAE": round(float(np.mean(results["test_mae"])), 4),
        }
        print(f"  [CollabFilter] CV RMSE={metrics['RMSE']}  MAE={metrics['MAE']}")
        return metrics

    def evaluate_precision_recall(
        self, ratings_df: pd.DataFrame, k: int = 10, threshold: float = 3.5
    ) -> dict:
        """
        Compute Precision@K and Recall@K on a held-out 20% test split.

        A movie is considered 'relevant' if its true rating ≥ threshold.
        """
        print(f"  [CollabFilter] Computing Precision@{k} and Recall@{k} …")
        reader = Reader(rating_scale=RATING_SCALE)
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader
        )
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        model_eval = SVD(n_factors=100, n_epochs=20, random_state=42)
        model_eval.fit(trainset)
        predictions = model_eval.test(testset)

        # Group predictions by user
        user_preds = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_preds[uid].append((est, true_r))

        precisions, recalls = [], []
        for uid, user_ratings in user_preds.items():
            # Sort by estimated rating, descending
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            top_k_preds = user_ratings[:k]

            # Number of relevant items in top-K
            n_rel_and_rec = sum(1 for _, true_r in top_k_preds if true_r >= threshold)
            n_rel_total = sum(1 for _, true_r in user_ratings if true_r >= threshold)
            n_rec = k

            precisions.append(n_rel_and_rec / n_rec if n_rec > 0 else 0)
            recalls.append(n_rel_and_rec / n_rel_total if n_rel_total > 0 else 0)

        metrics = {
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
        }
        print(f"  [CollabFilter] {metrics}")
        return metrics
