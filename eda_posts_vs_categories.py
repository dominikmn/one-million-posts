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

# %% [markdown]
# # Analyze comments vs article categories #2 
# Issue link: https://github.com/dominikmn/one-million-posts/issues/2

# %%
from utils import loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_posts = loading.load_extended_posts()

# %%
df_articles = loading.load_articles()

# %% [markdown]
# ## Data preparation

# %% [markdown]
# ### Encode post labels

# %%
cols_label = [c for c in df_posts.columns if c.startswith('label_')]

# %%
df_posts[cols_label] = df_posts[cols_label].replace({0.0:'no',1.0:'yes'})

# %% [markdown]
# ### Add path features

# %%
df_articles['path_split'] = df_articles['path'].str.split("/").apply(tuple)
df_articles['path_depth'] = df_articles['path_split'].apply(len)

# %%
df_articles.head()

# %%
df_posts['article_path_split'] = df_posts['article_path'].str.split("/").apply(tuple)
df_posts['article_path_depth'] = df_posts['article_path_split'].apply(len)

# %%
df_posts.head()

# %% [markdown]
# ## Articles per category

# %% [markdown]
# How many unique categories do we have?

# %%
df_articles['path_split'].nunique()

# %% [markdown]
# How many different paths are there per depth?

# %% tags=[]
df_articles[['path_split','path_depth']].drop_duplicates()['path_depth'].value_counts()\
    .reset_index().rename(columns={'index':'path_depth', 'path_depth':'count'})

# %% [markdown]
# What are the paths with depth 7 and 8?

# %%
df_articles[df_articles['path_depth'] >= 7]['path_split'].unique()

# %% [markdown]
# How many articles do we have per parent category?

# %%
df_articles['path_split'].apply(lambda x:x[0]).value_counts()

# %% [markdown]
# For which paths have the most articles been written?

# %%
df_articles['path_split'].value_counts()

# %% [markdown]
# ## Comments per category

# %% [markdown]
# How many _commments_ do we have per category?

# %%
df_posts['article_path_split'].value_counts().reset_index().head(10)

# %% [markdown]
# How many _comments_ do we have per parent category?

# %% tags=[]
df_posts['article_path_split'].apply(lambda x:x[0]).value_counts()

# %% [markdown]
# ## Comments vs Articles per path

# %% [markdown]
# How many articles and posts do we have per path?

# %%
cross_compare = df_posts.pivot_table(index='article_path', values=['id_post', 'id_article'], aggfunc='nunique')\
    .rename(columns={'id_article':'article_count', 'id_post':'post_count'})
cross_compare.sort_values(by=['article_count', 'post_count'], ascending=False, inplace=True)
cross_compare

# %%
d=cross_compare[['post_count','article_count']].head(20)
plot = plt.scatter(y=d.index, x=d['post_count'], c=d['post_count'], cmap='Reds')
plt.clf()
plt.colorbar(plot)
ax = sns.barplot(data=d, y=d.index, x=d['article_count'], hue='post_count', palette='Reds', dodge=False)
ax.set_ylabel('Categories')
ax.set_xlabel('Number of articles')
ax.legend_.remove()
ax.figure.set_size_inches(12,10)

# %%
# Calculate the percentage of both columns

#cross_compare['article_portion'] = cross_compare['article_count']/cross_compare['article_count'].sum()*100
#cross_compare['article_portion'] = cross_compare['article_portion'].round(3)
#cross_compare['post_portion'] = cross_compare['post_count']/cross_compare['post_count'].sum()*100
#cross_compare['post_portion'] = cross_compare['post_portion'].round(3)
#cross_compare

# %% [markdown]
# ## Comment-label per path

# %%
df_posts_reduced = df_posts.dropna(how='all', subset=cols_label)[['article_path']+cols_label]


# %%
def yes(x): return np.sum(x == 'yes')
def no(x):  return np.sum(x == 'no')
path_vs_label = df_posts_reduced.groupby('article_path').agg([no, yes])

# %%
sel1= [('label_inappropriate', 'yes'), ('label_discriminating', 'yes'), ('label_sentimentnegative', 'yes')]
sel2 = [(c, 'yes') for c in cols_label]

# %% [markdown]
# What are the top 'negative' categories?

# %%
path_vs_label[sel1].sort_values(by=sel1, ascending=False).head(10)

# %%
