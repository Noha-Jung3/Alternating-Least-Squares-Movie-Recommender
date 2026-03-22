import pandas as pd
def get_title(column, movie_dict, mov_index_from_column_to_global):
    """
    Get the movie title and categories for a local movie column index.

    Args:
        column: local column index (0-based)
        movie_dict: dictionary mapping global movieId -> (title, genres)
        mov_index_from_column_to_global: array mapping local column IDs to global IDs

    Returns:
        title: str
        categories: str
    """
    global_id = mov_index_from_column_to_global[column]  # map local column to global ID
    title, genres = movie_dict[global_id]               # lookup title and genres
    return title, genres