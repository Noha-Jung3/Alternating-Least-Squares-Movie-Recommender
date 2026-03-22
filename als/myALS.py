import numpy as np
from als.system_movie_j import system_movie_j
from als.system_user_i import system_user_i

def myALS(R, U, M, lambda_):
    """
    Perform ONE iteration of ALS.

    Args:
        R: ratings matrix (users x movies) — can be sparse or dense
        U: user feature matrix (f x n_users)
        M: movie feature matrix (f x n_movies)
        lambda_: regularization parameter

    Returns:
        Updated U and M
    """

    m, n = R.shape

    # --- Step 1: Update movie matrix M ---
    for j in range(n):
        Aj, bj = system_movie_j(R, U, lambda_, j)
        M[:, j] = np.linalg.solve(Aj, bj)

    # --- Step 2: Update user matrix U ---
    for i in range(m):
        Ai, bi = system_user_i(R, M, lambda_, i)
        U[:, i] = np.linalg.solve(Ai, bi)

    return U, M