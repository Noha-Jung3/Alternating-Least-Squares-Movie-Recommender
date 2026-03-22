def get_title(local_idx, titles_df, mov_index_from_column_to_global):
    """
    Get title and genres for a movie given its local column index.

    Args:
        local_idx: local column index (0-based)
        titles_df: pandas DataFrame with columns ['movieId', 'title', 'genres']
        mov_index_from_column_to_global: array mapping local index -> global movie ID

    Returns:
        title: string
        genres: string
    """
    global_id = mov_index_from_column_to_global[local_idx]

    # Find the row in titles_df that matches global_id
    row = titles_df[titles_df['movieId'] == global_id]
    if row.empty:
        return "Unknown", "Unknown"

    title = row['title'].values[0]
    genres = row['genres'].values[0]

    return title, genres