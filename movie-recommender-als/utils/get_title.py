import pandas as pd

def get_title(column, titles, mov_index_from_column_to_global):
    """
    Get the movie title and categories for a local movie column index.
    Args:
        column: local column index (0-based)
        titles: pandas DataFrame with movie info
        mov_index_from_column_to_global: array mapping local column IDs to global IDs
    Returns:
        title: str
        categories: str
    """
    globID = mov_index_from_column_to_global[column] + 1  # convert back to 1-based for titles
    row = titles.index[titles['movieId'] == globID].tolist()[0]
    title = titles.loc[row, 'title']
    categories = titles.loc[row, 'genres']
    return title, categories