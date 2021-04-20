import pandas as pd
from utils import feature_eng

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

df_disc = pd.read_csv('./output/trans_label_discriminating.csv')

df_disc = df_disc.fillna(value={"body": "", "headline": ""})
df_disc["orig"] = df_disc.headline + " " + df_disc.body
df_disc.orig = df_disc.orig.str.replace("\n", " ").str.replace("\r", " ")

df_disc.lang.nunique()

df_disc.lang.value_counts()

# languages: sk, pt-pt, nl, ru, es, da, ja, zh-Hans, oversample the rest

for i in df_disc.query('(lang=="km") or (lang=="tlh-Piqd") or (lang=="nl") or (lang=="da")').sort_values('id_post')[['text','lang','orig']].iloc[:24].iterrows():
    print(list(i[1]))
    print()
    print()


for i in df_disc.query('(lang=="zh-Hans") or (lang=="as") or (lang=="ru") or (lang=="de")').sort_values('id_post')[['text','lang','orig']].iloc[:24].iterrows():
    print(list(i[1]))
    print()
    print()



df_neg = pd.read_csv('./output/trans_label_sentimentnegative.csv')

df_neg= df_neg.fillna(value={"body": "", "headline": ""})
df_neg["orig"] = df_neg.headline + " " + df_neg.body
df_neg.orig = df_neg.orig.str.replace("\n", " ").str.replace("\r", " ")

df_neg.lang.value_counts()

# resample: en, ja

for i in df_disc.query('(lang=="kn") or (lang=="fil")').sort_values('id_post')[['text','lang','orig']].iloc[:12].iterrows():
    print(list(i[1]))
    print()
    print()


