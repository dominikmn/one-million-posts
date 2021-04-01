# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns

from utils import loading
from utils import feature_engineering

# %%
df = loading.load_posts()

# %%
df.head()

# %% [markdown]
# ## Get to know parent, leaf, and root posts

# %% tags=[]
# determine parent (comments that were commented) and leaf (comments that were not commented) posts
id_parent_posts = list(df.query("id_parent_post == id_parent_post").id_parent_post.unique())
id_leaf_posts = list(set(df.id_post).difference(df.id_parent_post))

# %%
# determine root (comments that comment an article) posts
id_root_posts = list(df.query("id_parent_post != id_parent_post").id_post)

id_root_leaf_posts = list(set(id_root_posts).intersection(id_leaf_posts))
id_root_parent_posts = list((set(id_root_posts).intersection(id_parent_posts)))

# %%
print(f"{'Number of parent posts: ':30s}{len(id_parent_posts)}")
print(f"{'Number of leaf posts: ':30s}{len(id_leaf_posts)}")
print(f"{'Sum of posts: ':30s}{len(id_parent_posts)+len(id_leaf_posts)}")
print(f"{'Shape of df_posts: ':30s}{df.shape}")
print("-"*20)
print(f"{'Number of root posts: ':30s}{len(id_root_posts)}")
print(f"{'Number of root-parent posts: ':30s}{len(id_root_parent_posts)}")
print(f"{'Number of root-leaf posts: ':30s}{len(id_root_leaf_posts)}")

# %% [markdown]
# ## Create new columns for analysis of tree structure

# %% tags=[]
df = feature_engineering.add_column_node_type(df)
df = feature_engineering.add_column_node_depth(df)
df = feature_engineering.add_column_number_subthreads(df)
df.head()

# %% [markdown]
# ## Analyze leaf/parent nodes per article
#
# ### How wide are discussion trees?
#
# The width of a discussion tree is defined by the number of discussion threads for this article, which is equal to the number of leaf nodes in the discussion tree.

# %%
# check number of leaf and parent nodes
df.node_type.value_counts()

# %%
discussion_width = df.query("node_type == 'leaf'").groupby("id_article").node_type.count()

# %%
discussion_width.describe()

# %%
df.id_article.nunique()

# %% [markdown]
# Discussion threads are on average 47 comments wide, whereas the median is at 13 comments. We have a minimum of one thread per article (in contrast to zero!) and a maximum of 2137 threads on the same article.

# %% [markdown]
# We want to analyze article discussions, therefore we want to know the discussion width for leaf nodes that have a node depth greater than one:

# %%
discussion_width = df.query("node_type == 'leaf' and node_depth > 1").groupby("id_article").node_type.count()
discussion_width.describe()

# %% [markdown]
# The width of discussion trees is on average 39.08 posts per article. The median is 11 threads, the minimum one, and the maximum 1783 threads.

# %% [markdown]
# Which article lead to 1783 separate threads?

# %%
discussion_width.sort_values()

# %%
df_articles = loading.load_articles()
df_articles.query("id_article == 3641")

# %% [markdown]
# The article 'Flüchtlingsthema katapultiert Strache auf Platz eins' has 1783 threads.

# %% [markdown]
# ### How long are discussion threads?
#
# The length of a discussion tree is defined by the number of posts between a leaf node and its root node.

# %% [markdown]
# We want to analyze article discussions, therefore we want to know the discussion length for leaf nodes that have a node depth greater than one:

# %%
df.query("node_type == 'leaf' and node_depth > 1").node_depth.describe()

# %% [markdown]
# The length of discussion trees in on average 3.19 posts per thread. The median is 3 posts per thread. The minimum is two post per thread and the maximum 62.

# %% [markdown]
# ### How long are discussion sub-threads?
#
# A subthread starts at any parent node that has two nodes referencing it.

# %%
df.query("number_subthreads > 0").number_subthreads.describe()

# %% [markdown]
# The average number of sub-threads starting at a post is 0.69. The miminum and the median is zero (not surprisingly, as there are more leaf posts than parent posts). The maximum is 31 posts referencing the same post.

# %%
