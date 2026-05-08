"""
eda.py  –  Exploratory Data Analysis
=====================================
Run this file to generate all EDA charts saved to  plots/
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from data_loader import get_dataset, rating_stats, top_movies, genre_distribution

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

movies, ratings, users = get_dataset()
rating_stats(ratings)

# ── 1. Rating Distribution ───────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Rating"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig("plots/01_rating_distribution.png", dpi=150); plt.close()

# ── 2. Ratings per User ──────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ratings.groupby("userId").size().hist(bins=50, ax=ax, color="#DD8452", edgecolor="white")
ax.set_title("Ratings per User", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of ratings"); ax.set_ylabel("Users")
plt.tight_layout(); plt.savefig("plots/02_ratings_per_user.png", dpi=150); plt.close()

# ── 3. Ratings per Movie ─────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ratings.groupby("movieId").size().hist(bins=50, ax=ax, color="#55A868", edgecolor="white")
ax.set_title("Ratings per Movie", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of ratings"); ax.set_ylabel("Movies")
plt.tight_layout(); plt.savefig("plots/03_ratings_per_movie.png", dpi=150); plt.close()

# ── 4. Top 10 Highest-Rated Movies ──────────
top = top_movies(ratings, movies, n=10)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=top, x="avg_rating", y="title", palette="Blues_d", ax=ax)
ax.set_title("Top 10 Movies (≥50 ratings)", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Rating"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("plots/04_top_movies.png", dpi=150); plt.close()

# ── 5. Genre Distribution ────────────────────
gdist = genre_distribution(movies)
fig, ax = plt.subplots(figsize=(9, 5))
gdist.sort_values().plot(kind="barh", ax=ax, color="#C44E52", edgecolor="white")
ax.set_title("Genre Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Movies")
plt.tight_layout(); plt.savefig("plots/05_genre_distribution.png", dpi=150); plt.close()

# ── 6. User Age Distribution ─────────────────
fig, ax = plt.subplots(figsize=(7, 4))
users["age"].hist(bins=20, ax=ax, color="#8172B2", edgecolor="white")
ax.set_title("User Age Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Users")
plt.tight_layout(); plt.savefig("plots/06_user_age.png", dpi=150); plt.close()

# ── 7. Rating Heatmap (sample) ───────────────
sample_users   = ratings["userId"].value_counts().head(20).index
sample_movies  = ratings["movieId"].value_counts().head(20).index
pivot = ratings[
    ratings["userId"].isin(sample_users) & ratings["movieId"].isin(sample_movies)
].pivot_table(index="userId", columns="movieId", values="rating")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Rating"})
ax.set_title("User-Movie Rating Matrix (Top 20×20)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/07_rating_heatmap.png", dpi=150); plt.close()

print("\n✅  All EDA plots saved to  plots/")
