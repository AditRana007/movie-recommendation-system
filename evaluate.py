"""
evaluate.py  –  Model Evaluation
==================================
Computes RMSE, MAE, Precision@K, Recall@K
No scikit-surprise required — pure NumPy/pandas/scikit-learn only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from data_loader import get_dataset
from recommender import CollaborativeRecommender, ContentBasedRecommender


# ── SVD K-Fold Cross-Validation ──────────────
def evaluate_svd_kfold(ratings_df: pd.DataFrame, k: int = 3):
    print(f"\n── SVD {k}-Fold Cross-Validation ──────────────────")
    kf     = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses, maes = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_df), 1):
        train = ratings_df.iloc[train_idx].reset_index(drop=True)
        test  = ratings_df.iloc[test_idx].reset_index(drop=True)

        cf = CollaborativeRecommender(n_factors=50)
        cf.fit(train)

        preds, actuals = [], []
        for row in test.itertuples(index=False):
            p = cf.predict(row.userId, row.movieId)
            preds.append(p)
            actuals.append(row.rating)

        preds   = np.array(preds)
        actuals = np.array(actuals)
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mae  = np.mean(np.abs(preds - actuals))
        rmses.append(rmse)
        maes.append(mae)
        print(f"  Fold {fold}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    print(f"\n  Mean RMSE : {np.mean(rmses):.4f}  +/- {np.std(rmses):.4f}")
    print(f"  Mean MAE  : {np.mean(maes):.4f}  +/- {np.std(maes):.4f}")


# ── Precision@K  /  Recall@K ─────────────────
def evaluate_top_k(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                   k: int = 10, threshold: float = 4.0):
    print(f"\n── Precision@{k} / Recall@{k}  (threshold={threshold}) ──────")

    train, test = train_test_split(ratings_df, test_size=0.2, random_state=42)

    cf = CollaborativeRecommender(n_factors=50)
    cf.fit(train)

    precisions, recalls = [], []
    for user_id in test["userId"].unique()[:100]:   # sample 100 users for speed
        relevant = set(
            test[(test["userId"] == user_id) & (test["rating"] >= threshold)]["movieId"]
        )
        if not relevant:
            continue

        recs = cf.recommend(user_id, movies_df, top_n=k)
        rec_ids = set(recs["movieId"])

        hit = len(rec_ids & relevant)
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    p  = np.mean(precisions)
    r  = np.mean(recalls)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  P@{k}={p:.4f}  R@{k}={r:.4f}  F1={f1:.4f}")


# ── Content-Based Coverage ───────────────────
def evaluate_content_coverage(movies_df: pd.DataFrame, sample: int = 50):
    cb = ContentBasedRecommender()
    cb.fit(movies_df)

    sample_titles = movies_df["title"].sample(sample, random_state=1).tolist()
    successes = sum(1 for t in sample_titles
                    if len(cb.recommend(t, top_n=10)) > 0)
    print(f"\n── Content-Based Coverage ─────────────────────────")
    print(f"  {successes}/{sample} titles returned results  ->  {successes/sample:.1%} coverage")


if __name__ == "__main__":
    movies, ratings, users = get_dataset()
    evaluate_svd_kfold(ratings, k=3)
    evaluate_top_k(ratings, movies, k=10)
    evaluate_content_coverage(movies)
