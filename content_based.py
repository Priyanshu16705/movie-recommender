"""
content_based.py
================
Content-Based Filtering using TF-IDF on movie genres + tags.

How it works:
  1. Build a TF-IDF matrix from the 'content' column of each movie.
  2. Compute pairwise Cosine Similarity between all movies.
  3. Given a query movie, return the top-K most similar movies.

The similarity matrix is saved to disk so it only needs to be built once.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class ContentBasedRecommender:
    """
    Recommends movies by computing cosine similarity on TF-IDF content vectors.

    Attributes
    ----------
    movie_df    : enriched movie DataFrame (from DataPreprocessor.build_movie_df)
    tfidf       : fitted TfidfVectorizer
    sim_matrix  : (n_movies × n_movies) cosine similarity ndarray
    title_index : Series mapping lower-case title → DataFrame index
    """

    def __init__(self, movie_df: pd.DataFrame):
        self.movie_df = movie_df.copy().reset_index(drop=True)
        self.tfidf: TfidfVectorizer | None = None
        self.sim_matrix: np.ndarray | None = None
        # Map lower-case title to integer index for fast lookup
        self.title_index = pd.Series(
            self.movie_df.index, index=self.movie_df["title"].str.lower()
        )

    # -------------------------------------------------------------------------
    # Training / building the similarity matrix
    # -------------------------------------------------------------------------

    def fit(self):
        """
        Fit TF-IDF on the 'content' column and compute the full cosine
        similarity matrix.  This is O(n²) in memory for n movies; for the
        small MovieLens dataset (~9k movies) it fits comfortably in RAM.
        """
        print("  [ContentBased] Fitting TF-IDF vectorizer …")
        self.tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),   # unigrams + bigrams for richer features
            min_df=1,
            stop_words="english",
        )
        tfidf_matrix = self.tfidf.fit_transform(self.movie_df["content"])
        print(f"  [ContentBased] TF-IDF matrix shape: {tfidf_matrix.shape}")

        print("  [ContentBased] Computing cosine similarity …")
        import numpy as np
        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float16)
        print(f"  [ContentBased] Similarity matrix shape: {self.sim_matrix.shape}")
        return self

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, models_dir: str = MODELS_DIR):
        """Persist the fitted vectorizer and similarity matrix."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(models_dir, "tfidf.pkl"))
        np.save(os.path.join(models_dir, "sim_matrix.npy"), self.sim_matrix)
        print(f"  [ContentBased] Models saved to {models_dir}")

    def load(self, models_dir: str = MODELS_DIR):
        """Load previously saved artifacts."""
        self.tfidf = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
        self.sim_matrix = np.load(os.path.join(models_dir, "sim_matrix.npy"))
        print(f"  [ContentBased] Models loaded from {models_dir}")
        return self

    # -------------------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------------------

    def recommend(
        self, title: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Return the top_k most similar movies to `title`.

        Parameters
        ----------
        title : str
            Partial or full movie title (case-insensitive substring match).
        top_k : int
            Number of recommendations to return.

        Returns
        -------
        DataFrame with columns [movieId, title, genres, avg_rating, similarity_score].
        Raises ValueError if no matching movie is found.
        """
        if self.sim_matrix is None:
            raise RuntimeError("Call fit() or load() before recommend().")

        # --- fuzzy title lookup -----------------------------------------------
        idx = self._find_index(title)

        # --- retrieve similarity scores for this movie -----------------------
        sim_scores = list(enumerate(self.sim_matrix[idx]))

        # Sort descending; skip the movie itself (score == 1.0 with itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_k]

        movie_indices = [i for i, _ in sim_scores]
        scores = [round(s, 4) for _, s in sim_scores]

        result = self.movie_df.loc[
            movie_indices, ["movieId", "title", "genres", "avg_rating"]
        ].copy()
        result["similarity_score"] = scores
        result = result.reset_index(drop=True)
        return result

    def _find_index(self, title: str) -> int:
        """
        Look up the DataFrame index for a movie title.
        Supports exact match (case-insensitive) and substring match.
        """
        query = title.lower().strip()

        # Exact match
        if query in self.title_index:
            return int(self.title_index[query])

        # Substring match – return the first hit
        matches = self.title_index[
            self.title_index.index.str.contains(query, na=False, regex=False)
        ]
        if len(matches) == 0:
            raise ValueError(
                f"Movie '{title}' not found in dataset. "
                "Try a different spelling or partial title."
            )
        return int(matches.iloc[0])

    def search_titles(self, query: str, max_results: int = 10) -> list[str]:
        """Return a list of movie titles that match the query string."""
        query = query.lower().strip()
        mask = self.movie_df["title"].str.lower().str.contains(query, na=False, regex=False)
        return self.movie_df.loc[mask, "title"].head(max_results).tolist()
