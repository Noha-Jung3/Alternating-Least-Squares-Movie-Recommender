import numpy as np
import scipy.sparse as sp

def system_user_i(R, M, lambda_, i):
    """
    Compute Ai and bi for updating user feature vector u_i.

    Args:
        R: ratings matrix (users x movies) — can be sparse or dense
        M: movie feature matrix (f x n_movies)
        lambda_: regularization parameter
        i: user index (0-based)

    Returns:
        Ai: (f x f) matrix
        bi: (f,) vector
    """

    # Movies rated by user i
    if sp.issparse(R):
        Ji = R[i, :].nonzero()[1]   # sparse: returns (row_indices, col_indices)
        q_Ji = R[i, Ji].toarray().flatten()
    else:
        Ji = np.nonzero(R[i, :])[0]
        q_Ji = R[i, Ji]

    nJi = len(Ji)
    f = M.shape[0]

    # Edge case: user hasn't rated anything
    if nJi == 0:
        return np.eye(f), np.zeros(f)

    # Submatrix of M for movies rated by user i
    M_Ji = M[:, Ji]

    # Normal equation components
    Ai = M_Ji @ M_Ji.T + lambda_ * nJi * np.eye(f)
    bi = M_Ji @ q_Ji

    return Ai, bi