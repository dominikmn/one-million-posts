import pandas as pd
import numpy as np

def calculate_top_words(series, relative=False):
    '''calculate word frequencies from a series of strings
    Arguments: series - a series containing strings, relative - If true calculate proportion of word frequency by number of observation, if False calculate absolute word frequency
    Return: topwords - a series containing word frequencies'''
    topwords=pd.Series(' '.join(series[(series==series)]).lower().split()).value_counts()
    if relative:
        return topwords/len(series)
    else:
        return topwords


def add_column_path_split(df, base_col='path', target_col='path_split', inplace=False):
    """
    Adds the column 'path_split' to a dataframe based on the column 'path'.
    Args: 
        dataframe: Pandas dataframe
        base_col: Name of the source column in that particular dataframe. Default is 'path'.
        target_col: Name of the new column. Default is 'path_split'.
        inplace: No effect yet
    Returns: 
        dataframe
    """
    assert base_col in df.columns
    df_new = df.copy()
    df_new[target_col] = df_new[base_col].str.split("/").apply(tuple)
    return df_new

def add_column_path_depth(df, base_col='path', target_col='path_depth', inplace=False):
    """
    Adds the column 'path_depth' to a dataframe based on the column 'path'.
    Args: 
        dataframe: Pandas dataframe
        base_col: Name of the source column in that particular dataframe. Default is 'path'.
        target_col: Name of the new column. Default is 'path_depth'.
        inplace: No effect yet
    Returns: 
        dataframe
    """
    assert base_col in df.columns
    df_new = df.copy()
    temp_col = 'path_split_temp'
    df_new = add_column_path_split(df_new, base_col=base_col, target_col=temp_col)
    df_new[target_col] = df_new[temp_col].apply(len)
    df_new.drop(columns=[temp_col], inplace=True)
    return df_new

def add_column_node_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `node_type` indicating whether a post is a parent or a leaf node

    Args:
        df: The posts DataFrame with the columns `id_post` and `id_parent_post`.

    Returns:
        df: A copy of df, extended by `node_type`.
    """
    if "node_type" not in df.columns:
        df_parent_posts = pd.DataFrame({"id_post": df.query("id_parent_post == id_parent_post").id_parent_post.unique()})
        df_parent_posts["node_type"] = "parent"

        return df.merge(df_parent_posts, how="left", on="id_post").replace({"node_type": np.nan}, "leaf")
    else:
        return df.copy()


def add_column_node_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `node_depth` stating the depth of the node up to this post.

    Args:
        df: The posts DataFrame with the columns `id_post` and `id_parent_post`.

    Returns:
        df: A copy of df, extended by `node_depth`.
    """
    df_out = df.copy()
    length = 0
    df_out["node_depth"] = length
    df_out.set_index(keys="id_post", inplace=True)
    next_nodes = df_out.query("id_parent_post != id_parent_post").index.to_list()
    while 0 in df_out.node_depth.unique():
        length += 1
        df_out.loc[next_nodes, "node_depth"] = length
        next_nodes = df_out.query("id_parent_post in @next_nodes").index.to_list()
    df_out.reset_index(inplace=True)
    return df_out


def add_column_number_subthreads(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `number_subthreads` stating the number of (sub-)threads that reference this post.

    Args:
        df: The posts DataFrame with the columns `id_post` and `id_parent_post`.

    Returns:
        df: A copy of df, extended by `number_subthreads`.
    """
    id_root_subthread = df.id_parent_post.value_counts()

    df_subthreads = id_root_subthread.reset_index().rename(columns={"index": "id_post", "id_parent_post": "number_subthreads"})
    df_subthreads.id_post = df_subthreads.id_post.astype(int)

    df_out = df.merge(df_subthreads, how="left", on="id_post")
    df_out.fillna({"number_subthreads": 0}, inplace=True)
    return df_out
