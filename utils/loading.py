# Imports
import pandas as pd
import sqlite3


def get_database_connection(path='./data/corpus.sqlite3'):
    con = sqlite3.connect(path)
    return con

def load_posts():
    con = get_database_connection()
    df_posts = pd.read_sql_query("select * from Posts", con)
    df_posts.columns = ["id_post", "id_parent_post", "id_article", "id_user", "created_at", "status", "headline", "body", "positive_votes", "negative_votes"]
    df_posts["created_at"] = pd.to_datetime(df_posts.created_at)
    return df_posts

def load_articles():
    con = get_database_connection()
    df_articles = pd.read_sql_query("select * from Articles", con)
    df_articles.columns = ['id_article', 'path', 'publishing_date_string', 'title', 'body']
    df_articles["publishing_date"] = pd.to_datetime(df_articles.publishing_date_string)
    return df_articles


def load_annotations():
    con = get_database_connection()
    df_annotations = pd.read_sql_query("select * from Annotations_consolidated", con)
    df_annotations.columns = df_annotations.columns.str.lower()
    return df_annotations

def load_pure_annotations():
    con = get_database_connection()
    df_annotations_pure = pd.read_sql_query("select * from Annotations", con)
    df_annotations_pure.columns = df_annotations_pure.columns.str.lower()
    return df_annotations_pure


def load_categories():
    con = get_database_connection()
    df_categories = pd.read_sql_query("select * from Categories", con)
    df_categories.columns = df_categories.columns.str.lower()
    return df_categories


def load_staff():
    con = get_database_connection()
    df_staff = pd.read_sql_query("select * from Newspaper_Staff", con)
    df_staff.columns = df_staff.columns.str.lower()
    return df_staff


def load_cv_split():
    con = get_database_connection()
    df_cv_split = pd.read_sql_query("select * from CrossValSplit", con)
    df_cv_split.columns = df_cv_split.columns.str.lower()
    return df_cv_split


def load_extended_posts():
    '''Load post table extended by annotations and staff.
    '''
    con = get_database_connection()
    df_annotations = load_annotations()
    df_posts = load_posts()
    df_staff = load_staff()
    df_articles = load_articles()

    # prepare annotations
    annotations = df_annotations.pivot(index="id_post", columns="category", values="value")
    annotations.columns = annotations.columns.str.lower()
    annotations = annotations.add_prefix("label_")

    # merge posts and annotations
    df = df_posts.merge(annotations, how="left", on="id_post")

    # add column `is_staff` indicating if posts is written by staff: yes-1, no-0
    id_staff = df_staff.id_user.to_list()
    df["is_staff"] = df.id_user.apply(lambda x: 1 if x in id_staff else 0)

    # add columns 'article_path' and 'article_title'    
    article_cats = df_articles[['id_article','path', 'title', 'publishing_date']]
    article_cats = article_cats.add_prefix('article_')
    df = df.merge(article_cats, how='left', left_on='id_article', right_on='article_id_article').drop('article_id_article', axis=1)
    
    return df
