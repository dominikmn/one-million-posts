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
# # Annotation analysis #25
# Issue link: https://github.com/dominikmn/one-million-posts/issues/25

# %%
import pandas as pd
import numpy as np
from utils import loading
from IPython.core.display import display, HTML

# %% [markdown]
# ## Annotator-Category uniqueness

# %%
df_annotations_pure = loading.load_pure_annotations()


# %%
def annotator_group (x): return tuple(x)
df_annotations_distinct = df_annotations_pure.groupby(["id_post","category"]).id_annotator.agg([annotator_group]).reset_index()
def category_group (x): return tuple(x)
df_annotations_distinct = df_annotations_distinct.groupby(["id_post","annotator_group"]).category.agg([category_group]).reset_index()

# %%
df_annotations_distinct.groupby("annotator_group").category_group.value_counts()

# %% [markdown]
# ## Overlapping sample of (1,2)
# There is one overlapping labeling of a sample by annotator 1 and 2.
# Let's have a look at the value.

# %%
instance_a = (1,2)
instance_p = df_annotations_distinct.query("annotator_group == @instance_a").id_post.iloc[0]
df_annotations_pure.query("id_post == @instance_p")

# %% [markdown]
# So the voting of this overlapping sample was luckily the same.

# %% [markdown]
# ## Round 3 analysis
# Round 3 picked posts from _specific_ articles in order to increase the class labels.
#
# Let's have a closer look on these articles.

# %%
df_articles = loading.load_articles()
df_posts = loading.load_extended_posts()


# %%
def give_articles_from_distinct(li_annotators, li_categories):
    posts = set(df_annotations_distinct.query("annotator_group in @li_annotators and category_group in @li_categories").id_post)
    print(f"Count of posts found for that combination: {len(posts)}")
    articles = set(df_posts[df_posts['id_post'].isin(posts)].id_article)
    print(f"Count of articles found for that combination: {len(articles)}")
    return articles


# %%
def print_articles(article_set):
    for i in article_set:
        print('-'*150)
        record = df_articles.query('id_article == @i')
        print(record.title.iloc[0])
        print('-'*150)
        display(HTML(record.body.iloc[0]))


# %% [markdown]
# ### Round 3 analysis - 'negative classes'
# The extra-labels for the 'negative classes' from round 3 were taken from "10 articles" according to the paper.
#
# It is worth to mention that theses posts have been actually labeled for **all** classes (not only the negative ones).
#
# Let's have a look on these articles.

# %%
li_a = [(1,), (2,), (4,)]
li_c = [('ArgumentsUsed', 'Discriminating', 'Inappropriate', 'OffTopic', 'PersonalStories', 'PossiblyFeedback', 'SentimentNegative', 'SentimentNeutral', 'SentimentPositive')]
articles_negative = give_articles_from_distinct(li_a,li_c)
print(articles_negative)

# %% [markdown]
# Indeed, we get 10 articles.
# The following dataframe shows them.
# * 3 articles deal with actions of **Israel towards Palestine**.
# * 4 articles deal with **immigrants/refugees in Austria and Greece**.
# * 3 articles deal with **violance against women or feminism/gender-roles**.

# %%
df_articles_negative = df_articles[df_articles['id_article'].isin(articles_negative)]
df_articles_negative

# %% [markdown]
# You can read the articles here:

# %%
articles_israelpalestine = {9767, 10820, 11105}
articles_refugees = {1860, 11004, 10425, 10707}
articles_women = {1172, 1704, 1831}
# Comment-in if you would like to read the articles
print_articles(articles_israelpalestine)
#print_articles(articles_refugees)
#print_articles(articles_women)

# %% [markdown]
# ### Round 3 analysis - 'personal stories'
# The extra-labels for personal stories from round 3 were taken from "13 discussions" according to the paper.
#
# Let's have a look on these articles.

# %%
li_a = [(1,), (2,), (4,)]
li_c = [('PersonalStories',)]
articles_discus = give_articles_from_distinct(li_a,li_c)
print(articles_discus)

# %% [markdown]
# So by 'discussions' the authors meant 'articles' in this case.
#
# The following dataframe shows them.
# We observe that 6 of the articles have been publishe from the time **before the June 2015**.

# %%
df_articles_discus = df_articles[df_articles['id_article'].isin(articles_discus)]
df_articles_discus

# %%
# Comment-in if you would like to read the articles
#print_articles(articles_discus)

# %%
