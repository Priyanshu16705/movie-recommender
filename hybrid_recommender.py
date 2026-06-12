"""
hybrid_recommender.py
=====================
Hybrid Recommendation: combines Content-Based and Collaborative Filtering scores.

Blending strategy
-----------------
  hybrid_score = α × content_similarity + (1 − α) × normalised_predicted_rating

  α (alpha) = 0.4 by default.  Increase α to lean more on content similarity;
  decrease it to rely more on personalised CF predictions.

Cold-start handling
-------------------
  When a user has no rating history (< 5 ratings), α is set to 1.0 so we fall
  back entirely to content-based recommendations.
"""

import numpy as np
import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


class HybridRecommender:
    """
    Blends content-based and collaborative filtering into a single ranked list.

    Parameters
    ----------
    content_rec : fitted ContentBasedRecommender
    cf_rec      : fitted CollaborativeFilteringRecommender
    alpha       : weight for content score  (0 = pure CF, 1 = pure content)
    """

    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        cf_rec: CollaborativeFilteringRecommender,
        alpha: float = 0.4,
    ):
        self.content_rec = content_rec
        self.cf_rec = cf_rec
        self.alpha = alpha

    # -------------------------------------------------------------------------
    # Core recommendation
    # -------------------------------------------------------------------------

    def recommend(
        self,
        title: str,
        user_id: int,
        ratings_df: pd.DataFrame,
        movie_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Return a ranked list of top_k hybrid recommendations.

        Parameters
        ----------
        title      : seed movie title (used for content similarity)
        user_id    : target user (used for CF personalisation)
        ratings_df : full ratings table
        movie_df   : enriched movie table
        top_k      : number of final recommendations

        Returns
        -------
        DataFrame [movieId, title, genres, avg_rating,
                   content_score, cf_score, hybrid_score]
        """
        # Detect cold-start; fall back to pure content if so
        is_cold = self.cf_rec.is_cold_start(user_id, ratings_df)
        effective_alpha = 1.0 if is_cold else self.alpha

        if is_cold:
            print(f"  [Hybrid] Cold-start user {user_id}: using content-only mode.")

        # ----- Step 1: get a large pool of candidates from content filter ----
        # We fetch 3× top_k candidates so the CF scores have enough to re-rank.
        pool_size = min(top_k * 3, len(movie_df) - 1)
        try:
            content_recs = self.content_rec.recommend(title, top_k=pool_size)
        except ValueError as e:
            raise ValueError(str(e))

        # ----- Step 2: add CF predicted ratings for each candidate -----------
        content_recs["cf_score_raw"] = content_recs["movieId"].apply(
            lambda mid: self.cf_rec.predict_rating(user_id, mid)
        )

        # ----- Step 3: normalise scores to [0, 1] ----------------------------
        content_recs["content_score"] = self._minmax(content_recs["similarity_score"])
        content_recs["cf_score"] = self._minmax(content_recs["cf_score_raw"])

        # ----- Step 4: blend -------------------------------------------------
        content_recs["hybrid_score"] = (
            effective_alpha * content_recs["content_score"]
            + (1 - effective_alpha) * content_recs["cf_score"]
        ).round(4)

        # ----- Step 5: sort and return top_k ---------------------------------
        result = (
            content_recs.sort_values("hybrid_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

        # Rename predicted rating column for clarity
        result = result.rename(columns={"cf_score_raw": "predicted_rating"})
        result["predicted_rating"] = result["predicted_rating"].round(2)

        return result[[
            "movieId", "title", "genres", "avg_rating",
            "content_score", "cf_score", "predicted_rating", "hybrid_score",
        ]]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        """Min-max normalise a pandas Series to [0, 1]."""
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - lo) / (hi - lo)
