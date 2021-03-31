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
# ## Preparation

# %% [markdown]
# ### Encode labels

# %%
cols_label = [c for c in df_posts.columns if c.startswith('label_')]

# %%
df_posts[cols_label] = df_posts[cols_label].replace({0.0:'no',1.0:'yes'})

# %%
df_posts.head(2)

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Parent categories

# %%
df_posts.article_path.str.split("/", n=1, expand=True).loc[:,0].value_counts()

# %% [markdown]
# ### Comments per path

# %%
df_posts['article_path'].value_counts()

# %% [markdown]
# ### Comments vs Articles per path

# %%
cross_compare = df_posts.pivot_table(index='article_path', values=['id_post', 'id_article'], aggfunc='nunique').rename(columns={'id_article':'article_count', 'id_post':'post_count'})
cross_compare.sort_values(by=['article_count', 'post_count'], ascending=False, inplace=True)
cross_compare

# %%
cross_compare['article_portion'] = cross_compare['article_count']/cross_compare['article_count'].sum()*100
cross_compare['article_portion'] = cross_compare['article_portion'].round(3)
cross_compare['post_portion'] = cross_compare['post_count']/cross_compare['post_count'].sum()*100
cross_compare['post_portion'] = cross_compare['post_portion'].round(3)
cross_compare

# %%
d=cross_compare[['post_portion','article_portion']].head(20)
plot = plt.scatter(y=d.index, x=d.post_portion, c=d.post_portion, cmap='Reds')
plt.clf()
plt.colorbar(plot)
ax = sns.barplot(data=d, y=d.index, x=d.article_portion, hue='post_portion', palette='Reds', dodge=False)
ax.set_ylabel('Categories')
ax.set_xlabel('% Percent of articles')
ax.legend_.remove()
ax.figure.set_size_inches(12,10)

# %% [markdown]
# ### Label per path

# %%
cols_label = [c for c in df_posts.columns if c.startswith('label_')]

# %%
df_posts_reduced = df_posts.dropna(how='all', subset=cols_label)[['article_path']+cols_label]


# %%
def yes(x): return np.sum(x == 'yes')
def no(x):  return np.sum(x == 'no')


# %%
path_vs_label = df_posts_reduced.groupby('article_path').agg([no, yes])

# %%
path_vs_label.tail(2)

# %%
sel1= [('label_inappropriate', 'yes'), ('label_discriminating', 'yes'), ('label_sentimentnegative', 'yes')]
sel2 = [(c, 'yes') for c in cols_label]

# %%
path_vs_label[sel1].sort_values(by=sel1, ascending=False).head(10)
