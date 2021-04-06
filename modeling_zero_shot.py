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
# #### Load useful librairies and data

# %%
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import loading, feature_engineering, scoring
from sklearn.metrics import precision_score, recall_score, f1_score

# %%
data = loading.load_extended_posts()

# %%
data = feature_engineering.add_column_ann_round(data)

# %%
data.head(5)

# %%
print(f'There are {data.shape[0]} rows in the dataset')

# %% [markdown]
# ### Preparing the pipeline in one-line of code!

# %%
#classifier = pipeline("zero-shot-classification",device = 0, model='joeddav/xlm-roberta-large-xnli')

# %% [markdown]
# ### Datapreparation

# %% [markdown]
# We concatenate the headline and the body of the text and replace `\n` and `\r`by whitespace.
#
# Then we create two seperate datasets. One containing the 1000 posts from the second annotation round and the other containing the 2599 posts from the third annotation round that were labled in all categories.

# %%
data.fillna(value={'headline':'', 'body':''}, inplace=True)
data['text'] = data['headline']+" "+data['body']
data['text']=data.text.str.replace('\n',' ').str.replace('\r', ' ')

# %%
data_1000 = data.query('ann_round==2').copy()
data_2599 = data.query('ann_round==3 & label_sentimentnegative==label_sentimentnegative')

# %% [markdown]
# ### Making predictions
#
# we map the labels from the dataset to the candidate labels we use in the zero-shot classification.
#
# We then predict the labels for both datasets and save the results to csv (*this is disabled because the predictions take several hours. Thus we load the saved csvs*).

# %%
category_map = {
                 'argumentation': 'label_argumentsused',
                 'discriminating': 'label_discriminating',
                 'inappropriate': 'label_inappropriate',
                 'a personal story': 'label_personalstories',
                 'off-topic': 'label_offtopic',
                 'requiring feedback': 'label_possiblyfeedback',
                 'positive': 'label_sentimentpositive',
                 'neutral': 'label_sentimentneutral',
                 'negative': 'label_sentimentnegative'
                }
candidate_labels = list(category_map.keys())

# %%
#predictedCategories_1000 = []
#for i in tqdm(range(1000)):
#    text = data_1000.iloc[i,]['text']
#    res = classifier(text, candidate_labels, multi_label=True)
#    labels = res['labels'] 
#    scores = res['scores'] #extracting the scores associated with the labels
#    res_dict = {category_map[label] : score for label,score in zip(labels, scores)}
#    res_dict['id_post'] = data_1000.iloc[i,]['id_post']
#    predictedCategories_1000.append(res_dict)

# %%
#data_pred_1000 = pd.DataFrame(predictedCategories_1000)

# %%
#data_pred_1000.to_csv('./output/zero_shot_result_1000.csv')

# %%
#predictedCategories_2599 = []
#for i in tqdm(range(2599)):
#    text = data_2599.iloc[i,]['text']
#    res = classifier(text, candidate_labels, multi_label=True)
#    labels = res['labels'] 
#    scores = res['scores'] #extracting the scores associated with the labels
#    res_dict = {category_map[label] : score for label,score in zip(labels, scores)}
#    res_dict['id_post'] = data_2599.iloc[i,]['id_post']
#    predictedCategories_2599.append(res_dict)

# %%
#data_pred_2599 = pd.DataFrame(predictedCategories_2599)

# %%
#data_pred_2599.to_csv('./output/zero_shot_result_2599.csv')

# %%
data_pred_1000 = pd.read_csv('./output/zero_shot_result_1000.csv', index_col=0)
data_pred_2599 = pd.read_csv('./output/zero_shot_result_2599.csv', index_col=0)

# %% [markdown]
# We now merge the predictions to the actual labels (and merge the datasets for round 2 and 3)

# %%
data_merge_1000 = pd.merge(data_1000, data_pred_1000, how='left', on = 'id_post', suffixes = ('_true', '_pred'))
data_merge_2599 = pd.merge(data_2599, data_pred_2599, how='left', on = 'id_post', suffixes = ('_true', '_pred'))
data_merge_3599 = pd.concat([data_merge_1000, data_merge_2599])

# %% [markdown]
# In the next step, we get the scoring for all datasets for a threshold of 0.5 and a threshold optimised on the f1-score and print in which categories the zero shot beat the baseline-model

# %%
scores_1000_05 = scoring.get_score_df(data_merge_1000)
scores_1000_best = scoring.get_score_df(data_merge_1000, best=True)
scores_2599_05 = scoring.get_score_df(data_merge_2599)
scores_2599_best = scoring.get_score_df(data_merge_2599, best=True)
scores_3599_05 = scoring.get_score_df(data_merge_3599)
scores_3599_best = scoring.get_score_df(data_merge_3599, best=True)

# %%
print('1000 posts, threshold = 0.5')
scoring.print_winners(scores_1000_05)
print('1000 posts, optimal threshold')
scoring.print_winners(scores_1000_best)
print('2599posts, threshold = 0.5')
scoring.print_winners(scores_2599_05)
print('2599 posts, optimal threshold')
scoring.print_winners(scores_2599_best)
print('3599posts, threshold = 0.5')
scoring.print_winners(scores_3599_05)
print('3599 posts, optimal threshold')
scoring.print_winners(scores_3599_best)
