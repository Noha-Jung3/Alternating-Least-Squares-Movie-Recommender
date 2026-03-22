import numpy as np
import scipy.sparse as sp
def system_movie_j(R, U, lambda_, j):
    """
    Compute Aj and bj to update movie feature vector m_j.
    Args:
        R: sparse ratings matrix (users x movies)
        U: user feature matrix (features x users)
        lambda_: regularization parameter
        j: movie index (0-based)
    Returns:
        Aj: f x f matrix
        bj: f-dimensional vector
    """
    
    Ij = R[:, j].nonzero()[0]
    
    # the number of users who ranked movie j
    nIj = len(Ij)
    
    # submatrix of U for those users
    U_Ij = U[:, Ij]
    
    # ratings for movie j
    r_Ij = R[Ij, j].toarray().flatten() if sp.issparse(R) else R[Ij, j]
    
    # normal equation (the matrix in they system for computing the optimal m_j)
    Aj = U_Ij @ U_Ij.T + lambda_ * nIj * np.eye(U.shape[0])
    # the RHS in the system for computing the optimal m_j
    bj = U_Ij @ r_Ij
    
    return Aj, bj


    