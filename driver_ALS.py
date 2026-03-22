import numpy as np
import matplotlib.pyplot as plt

from utils.movielens_load_data import load_movielens
from utils.get_title import get_title
from als.myALS import myALS
from als.gValue import gValue


def main():

    # --- Load data ---
    R, titles, mov_index_from_column_to_global = load_movielens(
        "data/ratings.csv",
        "data/movies.csv"
    )

    print("Data loaded.")
    print(f"Users: {R.shape[0]}, Movies: {R.shape[1]}")

    movie_dict = {
    row.movieId: (row.title, row.genres)
    for _, row in titles.iterrows()
    }

    # --- Parameters ---
    lambda_val = 0.02
    nits = 200
    f = 20

    nUsers, nMovies = R.shape

    # --- Initialise ---
    U = np.ones((f, nUsers))
    M = np.random.rand(f, nMovies)

    # --- Track objective values ---
    g_vals = []

    # --- ALS iterations ---
    for it in range(nits):

        U, M = myALS(R, U, M, lambda_val)

        g = gValue(R, U, M, lambda_val)
        g_vals.append(g)

        print(f"Iteration {it+1}/{nits}, g = {g:.4f}")

    # --- Plot convergence ---
    plt.figure()
    plt.plot(range(1, nits + 1), g_vals, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("g(U, M)")
    plt.title("ALS Convergence on MovieLens")
    plt.grid(True)

    # Save plot (for GitHub portfolio)
    plt.savefig("results/als_convergence.png")
    plt.show()

    # ===============================
    # 🎬 PART 1: Top 10 rated movies
    # ===============================

    user_id = 328  # Python is 0-indexed (MATLAB 329 → 328)

    if hasattr(R, "toarray"):
        user_ratings = R[user_id, :].toarray().flatten()
    else:
        user_ratings = R[user_id, :]

    rated_movie_ids = np.nonzero(user_ratings)[0]
    ratings = user_ratings[rated_movie_ids]

    # sort descending
    sorted_idx = np.argsort(ratings)[::-1]
    top10_ids = rated_movie_ids[sorted_idx[:10]]
    top10_ratings = ratings[sorted_idx[:10]]

    print("\n~ Top 10 movies rated highest by user 329 ~")
    for i, movie_id in enumerate(top10_ids, 1):
        title, genres = get_title(movie_id, titles, mov_index_from_column_to_global)
        print(f"{i}. {title} ({genres}) — Rating: {top10_ratings[i-1]:.1f}")

    # ===============================
    # 🎯 PART 2: Top 10 recommendations
    # ===============================

    # Predicted ratings
    predicted_scores = U[:, user_id].T @ M

    # Exclude already-rated movies
    predicted_scores[rated_movie_ids] = -np.inf

    # Top 10 recommendations
    rec_idx = np.argsort(predicted_scores)[::-1][:10]

    print("\n~ Top 10 recommended movies for user 329 ~")
    for i, movie_id in enumerate(rec_idx, 1):
        title, genres = get_title(movie_id, titles, mov_index_from_column_to_global)
        print(f"{i}. {title} ({genres}) — Predicted Rating: {predicted_scores[movie_id]:.2f}")


if __name__ == "__main__":
    main()