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
# # Plot f1-scores of several models #43
# Issue link: https://github.com/dominikmn/one-million-posts/issues/43

# %%
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import loading, feature_engineering, scoring
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import plotly.express as px
import matplotlib as plt

# %% [markdown]
# ## Scores Zero Shot

# %%
data = loading.load_extended_posts()
data = feature_engineering.add_column_ann_round(data)
data.fillna(value={'headline':'', 'body':''}, inplace=True)
data['text'] = data['headline']+" "+data['body']
data['text']=data.text.str.replace('\n',' ').str.replace('\r', ' ')
data_1000 = data.query('ann_round==2').copy()

# %%
data_pred_1000 = pd.read_csv('./output/zero_shot_result_1000.csv', index_col=0)
data_merge_1000 = pd.merge(data_1000, data_pred_1000, how='left', on = 'id_post', suffixes = ('_true', '_pred'))

# %%
scores_1000_05 = scoring.get_score_df(data_merge_1000)
scores_1000_best = scoring.get_score_df(data_merge_1000, best=True)

# %% tags=[]
y_col = [
        'label_sentimentnegative', 
        'label_sentimentpositive',
        'label_offtopic', 
        'label_inappropriate', 
        'label_discriminating', 
        'label_possiblyfeedback', 
        'label_personalstories', 
        'label_argumentsused',
    ]
#[scores_1000_05.query(f"label =='{l}'")['f1_pred'].iloc[0] for l in y_col]

# %% tags=[]
scores_1000_05

# %%
scores_zeroshot = scores_1000_05[['label', 'f1_pred']].query('label in @y_col').copy()
scores_zeroshot['model'] = pd.Series(['xlm-roberta-large-xnli']*9)
scores_zeroshot.rename(columns={'f1_pred':'f1_score'}, inplace=True)

# %%
scores_zeroshot

# %% [markdown]
# ## Scores MLflow

# %%
scores_mlflow = pd.read_csv('./output/mlflow_scores_21-04-09.csv', usecols=['label','val - F1','model',])

# %%
scores_mlflow.rename(columns={'val - F1':'f1_score'}, inplace=True)

# %% [markdown]
# ## Plotting the scores

# %%
scores_all = pd.concat([scores_zeroshot, scores_mlflow]).reset_index(drop=True) # reset_index because both frames bring an index 0,1,...8
scores_all.shape

# %%
scores_pivot = scores_all.pivot_table(index='label', columns='model', values='f1_score')#.reset_index()
#scores_pivot.loc[y_col[1],'BOW']
#scores_pivot.set_index('label')
scores_pivot

# %%
sns.heatmap(scores_pivot)

# %%
fig = px.imshow(scores_pivot)
fig.show()

# %%
