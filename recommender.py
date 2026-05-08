"""
Movie Recommendation System
============================
Techniques used:
  1. Content-Based Filtering  (TF-IDF + Cosine Similarity)
  2. Collaborative Filtering  (SVD via pure NumPy — no scikit-surprise needed)
  3. Hybrid Recommender       (weighted blend of both)

Dataset: MovieLens 100K
Dependencies: pandas, numpy, scikit-learn only
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """Recommend movies similar to a given title using TF-IDF on metadata."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.cosine_sim = None
        self.movies_df = None
        self.indices = None

    def fit(self, movies_df: pd.DataFrame):
        """
        movies_df must have columns: movieId, title, genres
        genres should be a pipe-separated string, e.g. "Action|Comedy|Drama"
        """
        self.movies_df = movies_df.reset_index(drop=True)
        # Replace pipe with space so TF-IDF treats each genre as a token
        self.movies_df["soup"] = (
            self.movies_df["title"].str.lower() + " " +
            self.movies_df["genres"].str.replace("|", " ", regex=False).str.lower()
        )
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df["soup"])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(
            self.movies_df.index, index=self.movies_df["title"]
        ).drop_duplicates()
        print(f"[ContentBased] Fitted on {len(self.movies_df)} movies.")

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return top-N movies most similar to 'title'."""
        if title not in self.indices:
            raise ValueError(f"Movie '{title}' not found in dataset.")
        val = self.indices[title]
        idx = int(val.iloc[0] if hasattr(val, "iloc") else val)
        sim_row = np.asarray(self.cosine_sim[idx]).flatten()
        sim_scores = sorted(enumerate(sim_row), key=lambda x: float(x[1]), reverse=True)
        sim_scores = sim_scores[1 : top_n + 1]   # exclude the movie itself
        movie_indices = [i[0] for i in sim_scores]
        result = self.movies_df.iloc[movie_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = [round(float(s[1]), 4) for s in sim_scores]
        return result.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING  (pure NumPy SVD)
# ─────────────────────────────────────────────

class CollaborativeRecommender:
    """
    Matrix Factorisation using numpy.linalg.svd — no external ML library needed.

    Steps:
      1. Build a User × Movie rating matrix (zeros for unseen)
      2. Mean-centre each user's ratings
      3. Truncate SVD to k latent factors
      4. Reconstruct the full matrix → predicted ratings
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors   = n_factors
        self.ratings_df  = None
        self.pred_matrix = None      # full reconstructed matrix
        self.user_index  = None      # userId  → row index
        self.movie_index = None      # movieId → col index
        self.movie_ids   = None      # col index → movieId
        self.user_mean   = None

    def fit(self, ratings_df: pd.DataFrame):
        """
        ratings_df must have columns: userId, movieId, rating
        """
        self.ratings_df = ratings_df.copy()

        # ── Build dense matrix ───────────────────
        user_ids  = sorted(ratings_df["userId"].unique())
        movie_ids = sorted(ratings_df["movieId"].unique())

        self.user_index  = {u: i for i, u in enumerate(user_ids)}
        self.movie_index = {m: j for j, m in enumerate(movie_ids)}
        self.movie_ids   = np.array(movie_ids)

        R = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
        for row in ratings_df.itertuples(index=False):
            i = self.user_index[row.userId]
            j = self.movie_index[row.movieId]
            R[i, j] = row.rating

        # ── Mean-centre rows (per-user mean, ignoring zeros) ─
        self.user_mean = np.true_divide(
            R.sum(axis=1),
            (R != 0).sum(axis=1).clip(min=1)
        )
        R_centred = R.copy()
        for i in range(R.shape[0]):
            R_centred[i, R[i] != 0] -= self.user_mean[i]

        # ── Truncated SVD ────────────────────────
        U, sigma, Vt = np.linalg.svd(R_centred, full_matrices=False)
        k  = min(self.n_factors, len(sigma))
        U  = U[:, :k]
        S  = np.diag(sigma[:k])
        Vt = Vt[:k, :]

        # Reconstruct + add back user mean
        R_hat = np.dot(U, np.dot(S, Vt))
        for i in range(R_hat.shape[0]):
            R_hat[i] += self.user_mean[i]

        self.pred_matrix = np.clip(R_hat, 1.0, 5.0)

        # ── Compute RMSE / MAE on observed ratings ──
        preds, actuals = [], []
        for row in ratings_df.itertuples(index=False):
            i = self.user_index[row.userId]
            j = self.movie_index[row.movieId]
            preds.append(self.pred_matrix[i, j])
            actuals.append(row.rating)
        preds   = np.array(preds)
        actuals = np.array(actuals)
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mae  = np.mean(np.abs(preds - actuals))
        print(f"[Collaborative] SVD fitted  |  k={k}  |  RMSE={rmse:.4f}  MAE={mae:.4f}")

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating of movie_id by user_id."""
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return 3.0   # fallback for unseen users/movies
        i = self.user_index[user_id]
        j = self.movie_index[movie_id]
        return float(self.pred_matrix[i, j])

    def recommend(self, user_id: int, movies_df: pd.DataFrame,
                  top_n: int = 10) -> pd.DataFrame:
        """Return top-N unrated movies predicted to be highest-rated by user."""
        rated = set(
            self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"]
        )
        if user_id not in self.user_index:
            print(f"[Collaborative] Unknown user {user_id}, returning popular movies.")
            popular = (
                self.ratings_df.groupby("movieId")["rating"]
                .mean().reset_index()
                .sort_values("rating", ascending=False)
            )
            top_ids = popular[~popular["movieId"].isin(rated)].head(top_n)["movieId"].tolist()
            result  = movies_df[movies_df["movieId"].isin(top_ids)].copy()
            score_map = dict(zip(popular["movieId"], popular["rating"]))
            result["predicted_rating"] = result["movieId"].map(score_map)
            return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

        i       = self.user_index[user_id]
        row     = self.pred_matrix[i]
        indices = np.argsort(row)[::-1]

        results = []
        for j in indices:
            mid = int(self.movie_ids[j])
            if mid not in rated:
                results.append((mid, float(row[j])))
            if len(results) == top_n:
                break

        result    = movies_df[movies_df["movieId"].isin([r[0] for r in results])].copy()
        score_map = {mid: score for mid, score in results}
        result["predicted_rating"] = result["movieId"].map(score_map)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. HYBRID RECOMMENDER
# ─────────────────────────────────────────────

class HybridRecommender:
    """
    Blend Content-Based and Collaborative scores with configurable weights.
    """

    def __init__(self, content_weight: float = 0.4, collab_weight: float = 0.6):
        assert abs(content_weight + collab_weight - 1.0) < 1e-6, \
            "Weights must sum to 1."
        self.content_weight = content_weight
        self.collab_weight  = collab_weight
        self.cb  = ContentBasedRecommender()
        self.cf  = CollaborativeRecommender()

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.cb.fit(movies_df)
        self.cf.fit(ratings_df)
        self.movies_df = movies_df

    def recommend(self, user_id: int, liked_title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Combine CB similarity scores with CF predicted ratings for a richer list.
        """
        scaler = MinMaxScaler()

        cb_recs = self.cb.recommend(liked_title, top_n=50)
        cb_recs = cb_recs.rename(columns={"similarity_score": "cb_score"})

        cf_recs = self.cf.recommend(user_id, self.movies_df, top_n=50)
        cf_recs = cf_recs.rename(columns={"predicted_rating": "cf_score"})

        merged = pd.merge(cb_recs, cf_recs[["movieId", "cf_score"]],
                          on="movieId", how="outer")
        merged["cb_score"] = merged["cb_score"].fillna(0)
        merged["cf_score"] = merged["cf_score"].fillna(merged["cf_score"].median())

        merged[["cb_norm"]] = scaler.fit_transform(merged[["cb_score"]])
        merged[["cf_norm"]] = scaler.fit_transform(merged[["cf_score"]])

        merged["hybrid_score"] = (
            self.content_weight * merged["cb_norm"] +
            self.collab_weight  * merged["cf_norm"]
        )
        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        return merged[["movieId", "title", "genres", "hybrid_score"]].reset_index(drop=True)