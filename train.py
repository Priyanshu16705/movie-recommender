"""
train.py
========
End-to-end training pipeline:
  1. Load & preprocess data.
  2. Fit Content-Based recommender (TF-IDF + cosine similarity).
  3. Fit Collaborative Filtering recommender (SVD).
  4. Evaluate both models and print metrics.
  5. Save all artefacts to models/.

Run from the project root:
    python train.py
"""

import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import load_and_prepare
from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def train():
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Load & Preprocess Data")
    print("=" * 60)
    movie_df, ratings_df, movies_raw = load_and_prepare()

    # ------------------------------------------------------------------
    # 2. Content-Based Filtering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Content-Based Filtering (TF-IDF + Cosine Similarity)")
    print("=" * 60)
    cb_rec = ContentBasedRecommender(movie_df)
    cb_rec.fit()
    cb_rec.save(MODELS_DIR)

    # Quick smoke test
    sample_title = movie_df["title"].iloc[0]
    sample_recs = cb_rec.recommend(sample_title, top_k=3)
    print(f"\n  Sample recommendations for '{sample_title}':")
    print(sample_recs[["title", "genres", "similarity_score"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Collaborative Filtering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Collaborative Filtering (SVD Matrix Factorisation)")
    print("=" * 60)
    cf_rec = CollaborativeFilteringRecommender()
    cf_rec.fit(ratings_df)
    cf_rec.save(MODELS_DIR)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)

    cv_metrics = cf_rec.evaluate_cv(ratings_df)
    pr_metrics = cf_rec.evaluate_precision_recall(ratings_df, k=10)

    all_metrics = {**cv_metrics, **pr_metrics}
    print("\n  ╔═══════════════════════════════╗")
    print("  ║       Evaluation Results      ║")
    print("  ╠═══════════════════════════════╣")
    for metric, value in all_metrics.items():
        print(f"  ║  {metric:<20s} {value:>6.4f}  ║")
    print("  ╚═══════════════════════════════╝")

    # Save metrics to JSON so the app can display them
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # Also save movie_df for quick loading in the app
    movie_df_path = os.path.join(MODELS_DIR, "movie_df.pkl")
    import joblib
    joblib.dump(movie_df, movie_df_path)
    print(f"  movie_df saved to {movie_df_path}")

    print("\n✅  Training complete!  Run: streamlit run app.py")


if __name__ == "__main__":
    train()
