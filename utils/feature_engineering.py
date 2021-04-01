import pandas as pd
import numpy as np

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