"""
app.py
======
Streamlit web application for the Movie Recommendation System.

Layout
------
Sidebar : settings (user ID, alpha, TMDB key, mode toggle)
Main    : search bar → recommendation cards with posters

Run with:
    streamlit run app.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from hybrid_recommender import HybridRecommender
from tmdb_api import TMDBClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "ml-latest-small")
# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for recommendation cards
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .rec-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid #44475a;
    }
    .rec-card h4 { color: #cba6f7; margin: 0 0 4px 0; font-size: 1rem; }
    .rec-card p  { color: #cdd6f4; margin: 2px 0; font-size: 0.82rem; }
    .badge {
        display: inline-block;
        background: #313244;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #89b4fa;
        margin: 2px 2px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    """Load all artefacts from disk once and cache in session."""
    movie_df  = joblib.load(os.path.join(MODELS_DIR, "movie_df.pkl"))
    print(f"Loaded {len(movie_df)} movies, sample: {movie_df['title'].iloc[0]}")
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    cb_rec = ContentBasedRecommender(movie_df)
    cb_rec.load(MODELS_DIR)

    cf_rec = CollaborativeFilteringRecommender()
    cf_rec.load(MODELS_DIR)

    return movie_df, ratings_df, cb_rec, cf_rec


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Helper: render a recommendation card
# ---------------------------------------------------------------------------
def render_card(row: pd.Series, tmdb: TMDBClient, show_scores: bool):
    """Render one movie as an HTML-styled card with a poster image."""
    info = tmdb.get_movie_info(row["title"])

    # Parse genres into badge-style chips
    genres = row.get("genres", "(no genres listed)")
    badges = "".join(
        f'<span class="badge">{g.strip()}</span>'
        for g in genres.split("|")
        if g.strip()
    )

    # Build score lines
    score_html = ""
    if show_scores:
        if "hybrid_score" in row and pd.notna(row.get("hybrid_score")):
            score_html += f'<p>🔀 Hybrid score: <b>{row["hybrid_score"]:.3f}</b></p>'
        if "predicted_rating" in row and pd.notna(row.get("predicted_rating")):
            score_html += f'<p>⭐ Predicted rating: <b>{row["predicted_rating"]:.2f} / 5.0</b></p>'
        if "similarity_score" in row and pd.notna(row.get("similarity_score")):
            score_html += f'<p>📐 Content similarity: <b>{row["similarity_score"]:.3f}</b></p>'

    avg_r = row.get("avg_rating", 0)
    avg_r_str = f"{avg_r:.2f}" if avg_r else "N/A"

    col_img, col_txt = st.columns([1, 3])
    with col_img:
        st.image(info["poster_url"], use_container_width=True)
    with col_txt:
        st.markdown(
            f"""
            <div class="rec-card">
              <h4>{row["title"]}</h4>
              <p>{badges}</p>
              <p>📊 Avg community rating: <b>{avg_r_str}</b></p>
              {score_html}
              <p style="margin-top:6px;color:#a6adc8;">{info["overview"][:220]}…</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(movie_df, ratings_df):
    st.sidebar.title("⚙️ Settings")

    # Read TMDB key: sidebar input overrides env/secrets
    default_tmdb = ""
    try:
        # On Streamlit Cloud, secrets are set via App Settings → Secrets
        default_tmdb = st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        import os
        default_tmdb = os.environ.get("TMDB_API_KEY", "")

    tmdb_key = st.sidebar.text_input(
        "TMDB API Key (optional)",
        value=default_tmdb,
        type="password",
        help="Get a free key at https://www.themoviedb.org/settings/api",
    )

    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Recommendation Mode",
        ["🔀 Hybrid", "🎭 Content-Based", "👥 Collaborative"],
        index=0,
    )

    user_id = st.sidebar.number_input(
        "User ID (for personalisation)",
        min_value=1,
        max_value=int(ratings_df["userId"].max()),
        value=1,
        step=1,
    )

    alpha = st.sidebar.slider(
        "Content weight α (Hybrid only)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="α=1 → pure content-based, α=0 → pure collaborative",
    )

    top_k = st.sidebar.slider("Top-K recommendations", 5, 20, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Model Metrics")
    metrics = load_metrics()
    if metrics:
        for k, v in metrics.items():
            st.sidebar.metric(k, v)
    else:
        st.sidebar.info("Run train.py to see metrics.")

    return tmdb_key, mode, int(user_id), alpha, top_k


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    st.title("🎬 Movie Recommendation System")
    st.caption("Content-Based · Collaborative Filtering · Hybrid  |  Powered by MovieLens + TMDB")

    # Guard: auto-train if models are missing (e.g. Streamlit Cloud cold start)
    if not os.path.exists(os.path.join(MODELS_DIR, "movie_df.pkl")):
        with st.spinner("⚙️  First-run setup: training models (60s)..."):
            try:
                import setup_models
                if not setup_models.data_exists():
                    setup_models.generate_synthetic_data()
                import train as _train
                _train.train()
                st.success("✅  Models trained! Reloading...")
                st.rerun()
            except Exception as e:
                st.error(f"Auto-setup failed: {e}. Run python train.py locally.")
                st.stop()

    # Load models
    movie_df, ratings_df, cb_rec, cf_rec = load_models()

    # Sidebar
    tmdb_key, mode, user_id, alpha, top_k = sidebar(movie_df, ratings_df)
    tmdb = TMDBClient(api_key=tmdb_key)

    # Search bar
    st.markdown("### 🔍 Search for a Movie")
    query = st.text_input(
        "Type a movie title …",
        placeholder="e.g. Toy Story, Inception, Dark Knight",
    )

    # Auto-complete suggestions
    if query:
        suggestions = cb_rec.search_titles(query, max_results=8)
        if not suggestions:
            st.warning("No movies found matching that title. Try a shorter term.")
            st.stop()

        selected_title = st.selectbox(
            "Select a movie to get recommendations for:",
            options=suggestions,
        )
    else:
        st.info("👆  Type a movie title above to get started.")
        st.stop()

    if st.button("🚀 Get Recommendations", type="primary"):
        st.markdown(f"---\n### 🎯 Top {top_k} Recommendations for **{selected_title}**")

        # Cold-start notice
        is_cold = cf_rec.is_cold_start(user_id, ratings_df)
        if is_cold and "Collaborative" in mode:
            st.warning(
                f"⚠️ User {user_id} has very few ratings — "
                "cold-start detected.  Falling back to content-based recommendations."
            )

        try:
            # --- Run the chosen recommender --------------------------------
            if "Content" in mode:
                with st.spinner("Computing content similarity …"):
                    recs = cb_rec.recommend(selected_title, top_k=top_k)
                show_scores = True

            elif "Collaborative" in mode:
                with st.spinner("Predicting ratings …"):
                    recs = cf_rec.recommend(user_id, ratings_df, movie_df, top_k=top_k)
                show_scores = True

            else:  # Hybrid
                hybrid = HybridRecommender(cb_rec, cf_rec, alpha=alpha)
                with st.spinner("Running hybrid model …"):
                    recs = hybrid.recommend(
                        selected_title, user_id, ratings_df, movie_df, top_k=top_k
                    )
                show_scores = True

        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        # --- Render cards --------------------------------------------------
        for _, row in recs.iterrows():
            render_card(row, tmdb, show_scores)
            st.markdown("")  # spacer

    # Footer
    st.markdown("---")
    st.caption(
        "Data: [MovieLens](https://grouplens.org/datasets/movielens/) · "
        "Posters: [TMDB](https://www.themoviedb.org/) · "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
