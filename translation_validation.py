# # Review validation sets and train sets #85
# Issue link: https://github.com/dominikmn/one-million-posts/issues/85

import pandas as pd

# ## Validation sets

df_val_disc = pd.read_csv('./output/trans_val_label_discriminating.csv')

df_val_disc = df_val_disc.fillna(value={"body": "", "headline": ""})
df_val_disc["orig"] = df_val_disc.headline + " " + df_val_disc.body
df_val_disc.orig = df_val_disc.orig.str.replace("\n", " ").str.replace("\r", " ")

for i in df_val_disc.sort_values('id_post')[['text','lang','orig']].iterrows():
    print(list(i[1]))
    print()
    print()


# Three positive samples, (back-)translated in six languages, swedish, spanish, polish, english, french and greek.
# The topics of the samples are:
# 1. Women
# 2. Migration
# 3. Turkey
#
# All translations are meaningful enough for our perspective.
# Variations: Sample 1 and 2 vary in all languages. Sample 3 is the same in all languages except for french.



df_val_ina = pd.read_csv('./output/trans_val_label_inappropriate.csv')

df_val_ina = df_val_ina.fillna(value={"body": "", "headline": ""})
df_val_ina["orig"] = df_val_ina.headline + " " + df_val_ina.body
df_val_ina.orig = df_val_ina.orig.str.replace("\n", " ").str.replace("\r", " ")

for i in df_val_ina.sort_values('id_post')[['text','lang','orig']].iterrows():
    print(list(i[1]))
    print()
    print()


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

# Conclusion:
#
# There are some inconsistencies and problems in the back-translations, but in general they seem to be okay. We accept them. Additional task: Add further positive examples from round 3.

# ## Train sets

import random

# ### Inaproppriate

df_train_ina = pd.read_csv('./output/trans_label_inappropriate.csv')

df_train_ina = df_train_ina.fillna(value={"body": "", "headline": ""})
df_train_ina["orig"] = df_train_ina.headline + " " + df_train_ina.body
df_train_ina.orig = df_train_ina.orig.str.replace("\n", " ").str.replace("\r", " ")

batch= random.randint(0,df_train_ina.shape[0]/7)
num_of_lang = 7
for n, i in enumerate(df_train_ina.sort_values('id_post')[['text','lang','orig']].iterrows()):
    if num_of_lang*batch < n < num_of_lang*(batch+1):
        print(list(i[1]))
        print()
        print()


# Languages used:
# * Filipino	fil -> not so well
# * Japanese	ja -> okay
# * German	de -> (obvious)
# * Maori 	mi -> Works well
# * Samoan	sm -> poor
# * Afrikaans	af (no clear tendency)
# * Kannada	kn -> good

Topics:
* Migration
* Insult

# ### Off-topic

df_train_off = pd.read_csv('./output/trans_label_offtopic.csv')

df_train_off = df_train_off.fillna(value={"body": "", "headline": ""})
df_train_off["orig"] = df_train_off.headline + " " + df_train_off.body
df_train_off.orig = df_train_off.orig.str.replace("\n", " ").str.replace("\r", " ")

num_of_lang = 5
batch= random.randint(0,df_train_off.shape[0]/num_of_lang)
for n, i in enumerate(df_train_off.sort_values('id_post')[['text','lang','orig']].iterrows()):
    if num_of_lang*batch < n < num_of_lang*(batch+1):
        print(list(i[1]))
        print()
        print()


# Languages used:
# * Filipino	fil -> not so well
# * Japanese	ja -> okay
# * Maori 	mi -> 
# * Samoan	sm -> very poor (even worse than ina)
# * Afrikaans	af
# * Kannada	kn -> good


