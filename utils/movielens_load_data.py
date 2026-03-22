import pandas as pd
import numpy as np
import scipy.sparse as sp

def load_movielens(ratings_csv, movies_csv):
    """
    Load MovieLens data into a sparse ratings matrix and mapping tables.

    Returns:
        R: sparse ratings matrix (users x movies) in CSR format
        titles: DataFrame
        mov_index_from_column_to_global: array mapping local column idx -> global movieId
    """
    ratings_df = pd.read_csv(ratings_csv)
    titles_df = pd.read_csv(movies_csv)

    # Users and movies
    user_ids = ratings_df['userId'].values
    movie_ids = ratings_df['movieId'].values
    rating_vals = ratings_df['rating'].values

    # Build sparse ratings matrix (users x movies)
    n_users = ratings_df['userId'].max()
    n_movies_global = ratings_df['movieId'].max()

    Rsp = sp.coo_matrix((rating_vals, (user_ids - 1, movie_ids - 1)), 
                        shape=(n_users, n_movies_global))

    # Convert to CSC to select columns (movies)
    Rsp = Rsp.tocsc()

    # Keep only columns (movies) that have at least 1 rating
    ind = np.array(Rsp.sum(axis=0)).flatten() > 0
    mov_index_from_column_to_global = np.where(ind)[0]  # 0-based indexing
    Rsp = Rsp[:, mov_index_from_column_to_global]

    # Convert back to CSR for efficient row operations in ALS
    return Rsp.tocsr(), titles_df, mov_index_from_column_to_global