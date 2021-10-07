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

# %% [markdown]
# # Review validation sets and train sets #85
# Issue link: https://github.com/dominikmn/one-million-posts/issues/85

# %%
import pandas as pd

# %% [markdown]
# ## Validation sets

# %%
df_val_disc = pd.read_csv('../output/trans_val_label_discriminating.csv')

# %%
df_val_disc = df_val_disc.fillna(value={"body": "", "headline": ""})
df_val_disc["orig"] = df_val_disc.headline + " " + df_val_disc.body
df_val_disc.orig = df_val_disc.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
for i in df_val_disc.sort_values('id_post')[['text','lang','orig']].iterrows():
    print(list(i[1]))
    print()
    print()


# %% [markdown]
# Three positive samples, (back-)translated in six languages, swedish, spanish, polish, english, french and greek.
# The topics of the samples are:
# 1. Women
# 2. Migration
# 3. Turkey
#
# All translations are meaningful enough for our perspective.
# Variations: Sample 1 and 2 vary in all languages. Sample 3 is the same in all languages except for french.

# %%

# %%
df_val_ina = pd.read_csv('../output/trans_val_label_inappropriate.csv')

# %%
df_val_ina = df_val_ina.fillna(value={"body": "", "headline": ""})
df_val_ina["orig"] = df_val_ina.headline + " " + df_val_ina.body
df_val_ina.orig = df_val_ina.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
for i in df_val_ina.sort_values('id_post')[['text','lang','orig']].iterrows():
    print(list(i[1]))
    print()
    print()


# %% [markdown]
# Sample 1: Topic Trump, politics. Meaning: not clear in orignal. Variation okay
# Sample 2: Topic homophobia, Meaning: No, variation: yes -> classification by chance
# Sample 3: Topic insult, Meaning: yes, variation, a bit
# Sample 4: Topic insult, meaning yes, variaton minimal
# Sample 5: Topic migration/turkey ... meaning yes, varation good
# Sample 6: Topic insult, meaning mixed, variation good
# Sample 7: Topic insult towards journalist, meaning good, variation good
# Sample 8: Topic insult towards newspaper, meaning mixed, variation good
# Sample 9: Topic insult towards other reader, meaning good, variation minimal, coloquial language
# Sample 10: Topic migration, meaning mostly good, variation good
#

# %% [markdown]
# Conclusion:
#
# There are some inconsistencies and problems in the back-translations, but in general they seem to be okay. We accept them. Additional task: Add further positive examples from round 3.

# %% [markdown]
# ## Train sets

# %%
import random

# %% [markdown]
# ### Inaproppriate

# %%
df_train_ina = pd.read_csv('../output/trans_label_inappropriate.csv')

# %%
df_train_ina = df_train_ina.fillna(value={"body": "", "headline": ""})
df_train_ina["orig"] = df_train_ina.headline + " " + df_train_ina.body
df_train_ina.orig = df_train_ina.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
batch= random.randint(0,df_train_ina.shape[0]/7)
num_of_lang = 7
for n, i in enumerate(df_train_ina.sort_values('id_post')[['text','lang','orig']].iterrows()):
    if num_of_lang*batch < n < num_of_lang*(batch+1):
        print(list(i[1]))
        print()
        print()


# %% [markdown]
# Languages used:
# * Filipino	fil -> not so well
# * Japanese	ja -> okay
# * German	de -> (obvious)
# * Maori 	mi -> Works well
# * Samoan	sm -> poor
# * Afrikaans	af (no clear tendency)
# * Kannada	kn -> good

# %% [markdown]
# Topics:
# * Migration
# * Insult

# %% [markdown]
# ### Off-topic

# %%
df_train_off = pd.read_csv('../output/trans_label_offtopic.csv')

# %%
df_train_off = df_train_off.fillna(value={"body": "", "headline": ""})
df_train_off["orig"] = df_train_off.headline + " " + df_train_off.body
df_train_off.orig = df_train_off.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
num_of_lang = 5
batch= random.randint(0,df_train_off.shape[0]/num_of_lang)
for n, i in enumerate(df_train_off.sort_values('id_post')[['text','lang','orig']].iterrows()):
    if num_of_lang*batch < n < num_of_lang*(batch+1):
        print(list(i[1]))
        print()
        print()


# %% [markdown]
# Languages used:
# * Filipino	fil -> not so well
# * Japanese	ja -> okay
# * Maori 	mi -> 
# * Samoan	sm -> very poor (even worse than ina)
# * Afrikaans	af
# * Kannada	kn -> good

# %% [markdown]
# ### Discriminating

# %%
df_disc = pd.read_csv('../output/trans_label_discriminating.csv')

# %%
df_disc = df_disc.fillna(value={"body": "", "headline": ""})
df_disc["orig"] = df_disc.headline + " " + df_disc.body
df_disc.orig = df_disc.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
df_disc.lang.nunique()

# %%
df_disc.lang.value_counts()

# %% [markdown]
# languages: sk, pt-pt, nl, ru, es, da, ja, zh-Hans, oversample the rest

# %%
for i in df_disc.query('(lang=="km") or (lang=="tlh-Piqd") or (lang=="nl") or (lang=="da")').sort_values('id_post')[['text','lang','orig']].iloc[:24].iterrows():
    print(list(i[1]))
    print()
    print()


# %%
for i in df_disc.query('(lang=="zh-Hans") or (lang=="as") or (lang=="ru") or (lang=="de")').sort_values('id_post')[['text','lang','orig']].iloc[:24].iterrows():
    print(list(i[1]))
    print()
    print()

# %% [markdown]
# ### Negative

# %%
df_neg = pd.read_csv('../output/trans_label_sentimentnegative.csv')

# %%
df_neg= df_neg.fillna(value={"body": "", "headline": ""})
df_neg["orig"] = df_neg.headline + " " + df_neg.body
df_neg.orig = df_neg.orig.str.replace("\n", " ").str.replace("\r", " ")

# %%
df_neg.lang.value_counts()

# %% [markdown]
# resample: en, ja

# %%
for i in df_disc.query('(lang=="kn") or (lang=="fil")').sort_values('id_post')[['text','lang','orig']].iloc[:12].iterrows():
    print(list(i[1]))
    print()
    print()

# %%
