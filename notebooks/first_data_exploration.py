# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
from IPython.core.display import display, HTML

# %%
con = sqlite3.connect('../data/corpus.sqlite3')
cur = con.cursor()

# %%
#cur.execute("select * from Annotations where instr(Body, ':-*') > 0")
cur.execute("select * from Annotations_consolidated limit 10")

# %% [markdown]
# # Vocabulary
#
# forum: All posts to one article?
#
# thread: ?
#
# First round: Dry run - Not in the dataset
#
# Second round: 1,000 posts (randomly selected) were annotated for each of the nine labels.
#
# Third round:
#
# 1. 2,599 posts from 10 articles were annotated for each of the nine labels (selected to increase frequency of negative sentiment, inappropriate, disciminating, and off-topic)
# 1. 5,737 posts were annotated regarding personal stories (selected from "share your thoughts" section)
# 1. 2,439 posts were annotated regarding feedback (selected from sample were moderators already answered or that were predicted as "needs feedback" by an existing model)
#
# A maximum of 1010 articles has at least one labled posts (for labels excluding feedback and personal stories (there may be more for these two))

# %% [markdown]
# ## Posts

# %%
df_posts = pd.read_sql_query("select * from Posts", con)
df_posts.columns = ["id_post", "id_parent_post", "id_article", "id_user", "created_at_string", "status", "headline", "body", "positive_votes", "negative_votes"]
df_posts["created_at"] = pd.to_datetime(df_posts.created_at_string)

# %%
df_posts

# %%
df_posts.describe(datetime_is_numeric=True)

# %% [markdown]
# There are posts with date before our time frame of the dataset. Let us investigate them further:

# %%
df_early_posts = df_posts.query("created_at < '2015-06-01'").copy()
df_early_posts

# %% [markdown]
# Are posts ordered by creation time? Therefore, what is the maximum id_post?

# %%
df_early_posts.describe().id_post

# %%
df_posts.loc[1842, "body"]

# %% [markdown]
# Posts after the official time frame of the dataset:

# %%
df_late_posts = df_posts.query("created_at > '2016-05-31'").copy()
df_late_posts

# %%
df_late_posts.id_article.nunique()

# %%
df_early_posts.id_post.hist()

# %%
df_late_posts.id_post.hist()

# %%
df_posts.info()

# %%
df_posts.status.unique()

# %%
df_posts.isnull().sum()

# %%
df_posts.query("body == ''").shape

# %%
df_posts.query("headline == ''").shape

# %%
df_posts.query("headline == '' and body == ''")

# %% [markdown]
# Check for posts where Body and Headline are empty string or None/NaN: \
# (`Headline != Headline` works because None/NaN is never None/NaN)

# %%
df_posts.query("(headline == '' or headline != headline) and (body == '' or body != body)")

# %%

# %%
df_posts[df_posts.body.isna()]

# %% [markdown]
# ## Articles

# %%
df_articles = pd.read_sql_query("select * from Articles", con)
df_articles.columns = ['id_article', 'path', 'publishing_date_string', 'title', 'body']
df_articles["publishing_date"] = pd.to_datetime(df_articles.publishing_date_string)
df_articles.head()

# %%
is_newsroom = df_articles.path.str.split("/", n=1, expand=True).loc[:,0]=="Newsroom"
df_articles[is_newsroom]

# %% [markdown]
# How many articles do we have per main category?

# %%
df_articles.path.str.split("/", n=1, expand=True).loc[:,0].value_counts()

# %% [markdown]
# What is Kiaroom???

# %%
is_kiaroom = df_articles.path.str.split("/", n=1, expand=True).loc[:,0] == "Kiaroom"
df_articles[is_kiaroom]

# %%
df_articles.describe(datetime_is_numeric=True)

# %% [markdown]
# ### Time on articles

# %%
df_early_articles = df_articles.query("publishing_date < '2015-06-01'").copy()
df_early_articles.head()

# %%
df_early_articles.shape

# %%
df_early_articles.id_article.nunique()

# %%

# %% [markdown]
# ## Annotations
#
# ### Consolidated

# %%
df_annotations = pd.read_sql_query("select * from Annotations_consolidated", con)
df_annotations.columns = df_annotations.columns.str.lower()
df_annotations.head()

# %%
df_annotations.describe()

# %% [markdown]
# ### Pure annotations

# %%
df_annotations_pure = pd.read_sql_query("select * from Annotations", con)
df_annotations_pure.columns = df_annotations_pure.columns.str.lower()
df_annotations_pure.head(20)

# %%
df_annotations_pure.groupby("id_annotator").category.value_counts()

# %% [markdown]
# Annotator 1 and 2 annotated in all rounds. Annotator 3 only annotated in the second round, whereas annotator 4 only annotated in round three.
#
# Checking annotations for round 3: Annotator 1: 2594-1000 = 1594, Annotator 2: 1513-1000 = 513, Annotator 4: 492; Overall annotations in round 3: 1594 + 513 + 492 = 2599

# %% [markdown]
# ## Categories

# %%
df_categories = pd.read_sql_query("select * from Categories", con)
df_categories.columns = df_categories.columns.str.lower()
df_categories

# %%
df_categories.shape

# %% [markdown]
# ## Newspaper staff

# %%
df_staff = pd.read_sql_query("select * from Newspaper_Staff", con)
df_staff.columns = df_staff.columns.str.lower()
df_staff

# %% [markdown]
# ## Cross-Val-Split
#
# Cross validation split of our annotations.

# %%
df_cv_split = pd.read_sql_query("select * from CrossValSplit", con)
df_cv_split.columns = df_cv_split.columns.str.lower()
df_cv_split

# %%
df_cv_split.id_post.nunique()

# %%
df_cv_split.category.value_counts()

# %%

# %% [markdown]
# ## Join annotation on post

# %%
annotations = df_annotations.pivot(index="id_post", columns="category", values="value")
annotations.columns = annotations.columns.str.lower()
annotations = annotations.add_prefix("label_")
annotations

# %%
df = df_posts.merge(annotations, how="left", on="id_post")
df.drop(columns=["created_at_string"], inplace=True)
df.head()

# %%
df.shape[0] == df_posts.shape[0]

# %%
df.info()

# %%
df_staff.head()

# %%
id_staff = df_staff.id_user.to_list()
df["is_staff"] = df.id_user.apply(lambda x: 1 if x in id_staff else 0)

# %%
df.head()

# %% [markdown]
# How many posts were written by staff?

# %%
df.is_staff.value_counts()

# %% [markdown]
# Do we have annotatons for posts by staff?

# %%
df.query("is_staff == 1")[annotations.columns].value_counts().reset_index().rename(columns={0: "count"})

# %% [markdown]
# There are two posts by staff that were both labeled as arguments used and sentiment neutral.

# %% [markdown]
# # Further experimentation

# %%
print(cur.fetchall())

# %%
df = pd.read_sql_query("select * from Annotations[v] where id_post = 3326", con)

# %%
df.head(100)

# %%
df = pd.read_sql_query("select * from Annotations_consolidated where id_post = 3326", con)

# %%
df.head(100)

# %%
pd.read_sql_query("select * from Posts where id_post == 3257", con).Body[0]

# %%

# %%
df_articles = pd.read_sql_query("Select * from Articles", con)

# %%
df_articles.head()

# %%
display(HTML(df_articles.Body[0]))

# %%
df_posts = pd.read_sql_query("Select * from Posts", con)

# %%
df_posts.head()

# %%
df_posts.query("ID_Article == 1")

# %%
