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
# # Plot f1-scores of several models #43
# Issue link: https://github.com/dominikmn/one-million-posts/issues/43

# %%
import pandas as pd
import numpy as np
from utils import loading, feature_engineering, scoring
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# %% [markdown]
# In order to use plotly inside jupyterlab follow these steps:
# 1. Install nodejs https://nodejs.org/en/download/package-manager/ 
# 2. install the jupyterlab-plotly extension
# ```
# jupyter labextension install jupyterlab-plotly@4.14.3
# ```

# %% [markdown]
# ## Label ordering

# %% tags=[]
# column order as in the paper Schabus,2017
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

# %% tags=[]
# column order grouped in [undesirable, desirable, neutral but requires action]
y_col_grouped = [
        'label_sentimentnegative', 
        'label_offtopic', 
        'label_inappropriate', 
        'label_discriminating', 
        'label_argumentsused',
        'label_personalstories', 
        'label_sentimentpositive',
        'label_possiblyfeedback', 
    ]

# %% tags=[]
# column order grouped in [undesirable, desirable, neutral but requires action]
y_col_grouped_clean= [
        'SentimentNegative', 
        'OffTopic', 
        'Inappropriate', 
        'Discriminating', 
        'ArgumentsUsed',
        'PersonalStories', 
        'SentimentPositive',
        'PossiblyFeedback', 
    ]

# %%
y_col_dict = {r:c for r,c in zip(y_col_grouped, y_col_grouped_clean)}

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
data_pred_1000 = pd.read_csv('../output/zero_shot_result_1000.csv', index_col=0)
data_merge_1000 = pd.merge(data_1000, data_pred_1000, how='left', on = 'id_post', suffixes = ('_true', '_pred'))
scores_1000_05 = scoring.get_score_df(data_merge_1000)
scores_1000_best = scoring.get_score_df(data_merge_1000, best=True)

# %%
scores_zeroshot = scores_1000_05[['label', 'f1_pred']].query('label in @y_col').copy()
scores_zeroshot['model'] = pd.Series(['xlm-roberta-large-xnli']*9)
scores_zeroshot.rename(columns={'f1_pred':'f1_score'}, inplace=True)

# %%
scores_zeroshot

# %% [markdown]
# ## Scores MLflow

# %%
scores_mlflow = pd.read_csv('../output/mlflow_scores_21-04-09.csv', usecols=['label','val - F1','model',])
scores_mlflow.rename(columns={'val - F1':'f1_score'}, inplace=True)
scores_mlflow.head(3)

# %% [markdown]
# ## Plotting the scores

# %%
scores_all = pd.concat([scores_zeroshot, scores_mlflow]).reset_index(drop=True) # reset_index because both frames bring an index 0,1,...8
scores_all.shape

# %%
scores_pivot = scores_all.pivot_table(index='label', columns='model', values='f1_score').reindex(y_col_grouped[::-1]) #reindex gives a custom order to the rows


# %%
scores = scores_pivot.iloc[:,[5,4,1,2]]  # choose only specific columns (i.e. models)

# %%
z      = np.array(scores.reset_index(drop=True)) # Must be a np.array instead of pandas DataFrame
z_text = [["{:.2f}".format(y) for y in x] for x in np.array(scores.reset_index(drop=True))]
x      = list(scores.columns) # Must be a list instead of pandas Series
y      = [y_col_dict[s] for s in scores.index] # Must be a list instead of pandas Series

# %%
layout_heatmap = go.Layout(
    #title=('Model f1 scores'),
    #xaxis=dict(title='Compared models'), 
    #yaxis=dict(title='Labels', dtick=1),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

ff_fig = ff.create_annotated_heatmap(
    z= z,
    x= x,
    y= y,
    annotation_text=z_text, 
    colorscale=[[0, '#d3d3d3'],[1, '#ec008e']],
    showscale = True,
    ygap=5,
    xgap=20,
)
fig  = go.FigureWidget(ff_fig)
fig.layout=layout_heatmap
fig.layout.annotations = ff_fig.layout.annotations
fig.data[0].colorbar = dict(title='F1 Score', titleside = 'right')
fig.update_layout(title_x=0.5)
fig.update_xaxes(side="top")

fig.show()

# %%
