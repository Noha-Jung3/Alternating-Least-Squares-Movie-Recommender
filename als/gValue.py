import numpy as np
import scipy.sparse as sp

def gValue(R, U, M, lambda_):
    """ 
    Compute the ALS objective function value.

    Args:
        R: ratings matrix (users x movies)
        U: user feature matrix (f x n_users)
        M: movie feature matrix (f x n_movies)
        lambda_: regularization parameter

    Returns:
        Scalar objective value
    """

    m,n = R.shape
    error_term = 0.0

    for j in range(n):
        if sp.issparse(R):
            Ij = R[:, j].nonzero()[0]
            r_Ij = R[Ij, j].toarray().flatten()
        else:
            Ij = np.nonzero(R[:, j])[0]
            r_Ij = R[Ij, j]

        if len(Ij) == 0:
            continue

        U_Ij = U[:, Ij]
        pred = U_Ij.T @ M[:, j]

        error_term += np.sum((r_Ij - pred) ** 2)

    # --- Regularisation for U ---
    reg_term_u = 0.0
    for i in range(m):
        if sp.issparse(R):
            count = R[i, :].count_nonzero()
        else:
            count = np.count_nonzero(R[i, :])

        reg_term_u += count * np.linalg.norm(U[:, i])**2

    # --- Regularisation for M ---
    reg_term_m = 0.0
    for j in range(n):
        if sp.issparse(R):
            count = R[:, j].count_nonzero()
        else:
            count = np.count_nonzero(R[:, j])

        reg_term_m += count * np.linalg.norm(M[:, j])**2

    return error_term + lambda_ * (reg_term_u + reg_term_m)
    
