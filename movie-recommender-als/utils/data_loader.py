import pandas as pd
import numpy as np
import scipy.sparse as sp

def load_movielens(ratings_file, movies_file):
    """Load MovieLens ratings and movie titles into a sparse matrix and supporting arrays Returns:
        Rsp: sparse ratings matrix (users x movies)
        titles: pandas DataFrame with movie info
        mov_index_from_column_to_global: maps local column IDs to global movie IDs
    """
    #Load ratings 
    ratings = pd.read_csv(ratings_file)
    #columns: userId, movieId, rating, timestamp

    #Load movie titles:
    titles = pd.read_csv(movies_file)
    # columns: movieID, title, genres 

    #Build sparse ratings matrix 
    n_users = ratings["UserId"].max()
    n_movies_global = ratings["MovieId"].max()

    #Initial sparse matrix with global movie IDs:
    Rsp_full = sp.lil_matrix((n_users, n_movies_global))
    for row in ratings.itertuples(index=False):
        Rsp_full[row.userId - 1, row.movieId - 1] = row.rating

    #Identify movies with at least one rating
    nonempty_cols = np.array(Rsp_full.sum(axis=0)).flatten() > 0
    mov_index_from_column_to_global = np.where(nonempty_cols)[0]

    # Select only non-empty columns
    Rsp = Rsp_full[:, mov_index_from_column_to_global].tocsc()
    
    return Rsp, titles, mov_index_from_column_to_global


