"""
tmdb_api.py
===========
Thin wrapper around the TMDB (The Movie Database) API.

Usage
-----
    from tmdb_api import TMDBClient
    client = TMDBClient(api_key="YOUR_KEY")
    poster_url = client.get_poster_url("Inception")

If no API key is provided, or if the request fails, the client returns
a placeholder image URL so the UI degrades gracefully.
"""

import os
import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE   = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Poster"


class TMDBClient:
    """
    Fetches movie posters and metadata from TMDB.

    Parameters
    ----------
    api_key : str or None
        TMDB API key.  If None, falls back to the TMDB_API_KEY env variable.
        If neither is set, all requests return placeholder images.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TMDB_API_KEY", "")

    def _search(self, title: str) -> dict | None:
        """Hit the TMDB search endpoint and return the first result dict."""
        if not self.api_key:
            return None
        # Strip year in parentheses, e.g. "Toy Story (1995)" → "Toy Story"
        clean_title = title.split("(")[0].strip()
        try:
            resp = requests.get(
                TMDB_SEARCH_URL,
                params={"api_key": self.api_key, "query": clean_title},
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return results[0] if results else None
        except Exception:
            return None

    def get_poster_url(self, title: str) -> str:
        """
        Return a fully qualified TMDB poster URL for the given movie title.
        Falls back to a placeholder on any error.
        """
        result = self._search(title)
        if result and result.get("poster_path"):
            return TMDB_IMG_BASE + result["poster_path"]
        return PLACEHOLDER_IMG

    def get_movie_info(self, title: str) -> dict:
        """
        Return a dict with keys: poster_url, overview, release_date, vote_average.
        All fields fall back to safe defaults on error.
        """
        result = self._search(title)
        if result is None:
            return {
                "poster_url": PLACEHOLDER_IMG,
                "overview": "No overview available.",
                "release_date": "Unknown",
                "vote_average": "N/A",
            }
        poster_path = result.get("poster_path")
        return {
            "poster_url": (TMDB_IMG_BASE + poster_path) if poster_path else PLACEHOLDER_IMG,
            "overview": result.get("overview", "No overview available."),
            "release_date": result.get("release_date", "Unknown"),
            "vote_average": result.get("vote_average", "N/A"),
        }
