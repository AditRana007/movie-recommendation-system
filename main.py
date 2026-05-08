"""
main.py  –  Full Pipeline Demo
================================
Run this file to see all three recommenders in action.

  python main.py
"""

from data_loader import get_dataset
from recommender  import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender


def banner(text: str):
    w = 60
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main():
    # ── 1. Load Data ────────────────────────────
    banner("1 / 4   Loading Dataset")
    movies, ratings, users = get_dataset()
    print(movies.head(3).to_string(index=False))

    # ── 2. Content-Based ────────────────────────
    banner("2 / 4   Content-Based Filtering")
    cb = ContentBasedRecommender()
    cb.fit(movies)

    QUERY_TITLE = "Toy Story (1995)"
    print(f"\nMovies similar to  '{QUERY_TITLE}':\n")
    cb_recs = cb.recommend(QUERY_TITLE, top_n=10)
    print(cb_recs.to_string(index=False))

    # ── 3. Collaborative Filtering ──────────────
    banner("3 / 4   Collaborative Filtering (SVD)")
    cf = CollaborativeRecommender(n_factors=50)
    cf.fit(ratings)

    USER_ID = 42
    print(f"\nTop recommendations for  User {USER_ID}:\n")
    cf_recs = cf.recommend(USER_ID, movies, top_n=10)
    print(cf_recs.to_string(index=False))

    # ── 4. Hybrid Recommender ───────────────────
    banner("4 / 4   Hybrid Recommender (CB 40% + CF 60%)")
    hybrid = HybridRecommender(content_weight=0.4, collab_weight=0.6)
    hybrid.fit(movies, ratings)

    print(f"\nHybrid recommendations for  User {USER_ID}  based on  '{QUERY_TITLE}':\n")
    h_recs = hybrid.recommend(USER_ID, QUERY_TITLE, top_n=10)
    print(h_recs.to_string(index=False))

    banner("Done!  Run  eda.py  for visualisations  |  evaluate.py  for metrics")


if __name__ == "__main__":
    main()
