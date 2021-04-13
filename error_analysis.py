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
# # Error analysis - Modeling Round 1
#
# [Issue 49](https://github.com/dominikmn/one-million-posts/issues/49)
#
# ## Results
#
# The first error analysis includes posts that were misclassified by the NaiveBayes model for the labels "inappropriate", "offtopic", and "argumentsused". These posts were annotated regarding "uses coloquial language", "includes url", and "has typos".
#
# **The analysed patterns did not suggest a systematic error in classification.**
#
# Furthermore, all posts of the validation set were grouped by the number of misclassifications in all labels. Posts got misclassified in maximal 5 of the 7 labels. The posts were then analysed for differences based on the number of misclassifications.
#
# Interesting observations about the group of posts that were always classified correctly:
#     
# + no one word posts
# + includes posts with only mathematical calculation
# + tend to have longer text
#
# **!!! Important Note !!!**: The train-test-validation split got an update after the first modeling round and is not equivalent to the here analysed posts any longer. Furthermore, the models need to be trained locally to predict on the validation set using `modeling.predict_error_analysis.py`. The thereby saved .csv-files are used for this analysis.

# %% [markdown]
# ## Helpers

# %%
import pandas as pd
import glob

from utils import loading, feature_engineering

# %%
df_posts = loading.load_extended_posts(split="val")
df_posts = feature_engineering.add_column_text(df_posts)


# %%
def print_posts(df):
    def __print_post(row):
        print("-"*20)
        print(f"ID Post: {row.id_post}")
        print(f"{row.text}")
    _ = df.apply(lambda x: __print_post(x), axis=1)


# %% [markdown]
# ## NaiveBayes_label_inappropriate_2021-04-08_201325
#
# ```bash
# python -m modeling.predict models/NaiveBayes_label_inappropriate_2021-04-08_201325 label_inappropriate
# ```
#
# ### False Negatives

# %% tags=[]
df_fn_ids = pd.read_csv("./output/misclassification_NaiveBayes_label_inappropriate_2021-04-08_201325_fn.csv", index_col=[0])

# %%
df_fn = df_posts.merge(df_fn_ids, on="id_post", how="inner")[["id_post", "text"]]
print_posts(df_fn)

# %% [markdown]
# ### False Positives

# %%
df_fp_ids = pd.read_csv("./output/misclassification_NaiveBayes_label_inappropriate_2021-04-08_201325_fp.csv", index_col=[0])

# %%
df_fp = df_posts.merge(df_fp_ids, on="id_post", how="inner")[["id_post", "text"]]
print_posts(df_fp)

# %% [markdown]
# ## NaiveBayes_label_offtopic_2021-04-08_201346
#
# ```bash
# python -m modeling.predict models/NaiveBayes_label_offtopic_2021-04-08_201346 label_offtopic
# ```
#
# ### False Negatives

# %% tags=[]
df_fn_ids = pd.read_csv("./output/misclassification_NaiveBayes_label_offtopic_2021-04-08_201346_fn.csv", index_col=[0])

# %%
df_fn = df_posts.merge(df_fn_ids, on="id_post", how="inner")[["id_post", "text"]]
print_posts(df_fn)

# %% [markdown]
# ### False Positives

# %%
df_fp_ids = pd.read_csv("./output/misclassification_NaiveBayes_label_offtopic_2021-04-08_201346_fp.csv", index_col=[0])

# %%
df_fp = df_posts.merge(df_fp_ids, on="id_post", how="inner")[["id_post", "text"]]
print_posts(df_fp)

# %% [markdown]
# ## NaiveBayes_label_argumentsused_2021-04-08_201243
#
# ```bash
# python -m modeling.predict models/NaiveBayes_label_argumentsused_2021-04-08_201243 label_argumentsused
# ```

# %% [markdown]
# ### False Negatives

# %%
df_fn = pd.read_csv("./output/misclassification_NaiveBayes_label_argumentsused_2021-04-08_201243_fn.csv", index_col=[0])

# %%
df_fn.head()

# %%
df_posts.query("id_post == 821599").text.to_list()[0]

# %% [markdown]
# ### False Positives

# %%
df_fp = pd.read_csv("./output/misclassification_NaiveBayes_label_argumentsused_2021-04-08_201243_fp.csv", index_col=[0])

# %%
df = df_posts.merge(df_fp, on="id_post", how="inner")[["id_post", "text"]]
print_posts(df)

# %% [markdown]
# ## Analysis of misclassifications across labels

# %%
# Load all predictions and add to df
rdir = r"./output"
flist = glob.glob(os.path.join(rdir,"*_pred.csv"))

df = loading.load_extended_posts()
df = feature_engineering.add_column_ann_round(df)
df = feature_engineering.add_column_text(df)
df = df.query("ann_round == 2")
for f in flist:
    df_temp = pd.read_csv(f, index_col=[0])    
    df = pd.merge(df, df_temp, on="id_post", how="right")

# %%
# Calculate number of labels for which a post was misclassified
labels = ['label_argumentsused', 'label_discriminating', 'label_inappropriate', 'label_offtopic', 'label_personalstories', 'label_possiblyfeedback', 'label_sentimentnegative', 'label_sentimentneutral', 'label_sentimentpositive']
df["n_miss"] = 0
for label in labels:
    label_pred = f"{label}_pred"
    df["n_miss"] += abs(df[label] - df[label_pred])

# %% [markdown]
# How many posts do we have for each amount of misclassified labels?

# %%
df.n_miss.value_counts()

# %% [markdown]
# Are there differences in text length for the different amounts of misclassified labels?

# %%
df["n_words"] = df.text.str.strip().str.split(r' +').apply(lambda x: len(x))

df.groupby("n_miss").n_words.describe()

# %% [markdown]
# Posts that were correctly classified for all labels seem to have longer texts.

# %% [markdown]
# #### Print all comments based on their number of misclassifications

# %%
print_posts(df.query("n_miss == 5"))

# %%
print_posts(df.query("n_miss == 4"))

# %%
print_posts(df.query("n_miss == 3"))

# %%
print_posts(df.query("n_miss == 2"))

# %%
print_posts(df.query("n_miss == 1"))

# %%
print_posts(df.query("n_miss == 0"))

# %%

# %%
