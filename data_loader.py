"""
data_loader.py
==============
Download and preprocess the MovieLens 100K dataset.
If the files already exist locally, they are loaded from disk.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR      = "data/ml-100k"


def download_movielens(force: bool = False):
    """Download and unzip MovieLens 100K if not already present."""
    zip_path = "data/ml-100k.zip"
    if not os.path.exists(DATA_DIR) or force:
        os.makedirs("data", exist_ok=True)
        print("Downloading MovieLens 100K …")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data/")
        os.remove(zip_path)
        print("Download complete.")
    else:
        print("Dataset already exists, skipping download.")


def load_movies() -> pd.DataFrame:
    """Load movie metadata (movieId, title, genres)."""
    path = os.path.join(DATA_DIR, "u.item")
    cols = ["movieId", "title", "release_date", "video_release_date",
            "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western"]
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")

    # Convert binary genre columns → pipe-separated string
    genre_cols = cols[6:]
    df["genres"] = df[genre_cols].apply(
        lambda row: "|".join([g for g, v in zip(genre_cols, row) if v == 1]),
        axis=1
    )
    df["genres"] = df["genres"].replace("", "Unknown")

    # Clean title: remove year in parentheses for display
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()

    return df[["movieId", "title", "title_clean", "genres"]].copy()


def load_ratings() -> pd.DataFrame:
    """Load user ratings (userId, movieId, rating, timestamp)."""
    path = os.path.join(DATA_DIR, "u.data")
    df = pd.read_csv(path, sep="\t",
                     names=["userId", "movieId", "rating", "timestamp"])
    # Convert integer ratings (1-5) to half-star scale expected by Surprise
    df["rating"] = df["rating"].astype(float)
    return df


def load_users() -> pd.DataFrame:
    """Load user demographic data."""
    path = os.path.join(DATA_DIR, "u.user")
    df = pd.read_csv(path, sep="|",
                     names=["userId", "age", "gender", "occupation", "zip_code"])
    return df


def get_dataset():
    """
    Convenience function: download (if needed) and return
    (movies_df, ratings_df, users_df) as a tuple.
    """
    download_movielens()
    movies  = load_movies()
    ratings = load_ratings()
    users   = load_users()
    print(f"Movies : {len(movies):,}   |   Ratings : {len(ratings):,}   |   Users : {len(users):,}")
    return movies, ratings, users


# ─────────────────────────────────────────────
# EDA  helpers
# ─────────────────────────────────────────────

def rating_stats(ratings_df: pd.DataFrame):
    print("\n── Rating Statistics ──────────────────")
    print(ratings_df["rating"].describe().to_string())
    print(f"\nSparsity : {1 - len(ratings_df) / (ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()):.4%}")


def top_movies(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n: int = 10):
    avg = (
        ratings_df.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
    )
    avg = avg[avg["num_ratings"] >= 50]          # minimum vote threshold
    avg = avg.merge(movies_df[["movieId", "title"]], on="movieId")
    return avg.sort_values("avg_rating", ascending=False).head(n)


def genre_distribution(movies_df: pd.DataFrame) -> pd.Series:
    all_genres = movies_df["genres"].str.split("|").explode()
    return all_genres.value_counts()
