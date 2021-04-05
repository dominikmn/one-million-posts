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

# %% tags=[]
# !pip install sentencepiece

# %%
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import loading, feature_engineering

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
classifier = pipeline("zero-shot-classification",device = 0, model='joeddav/xlm-roberta-large-xnli')

# %% [markdown]
# ### Making Predictions

# %% [markdown]
# This model works best with informative labels, spam/ham ar not so inforamtive. Using spam/ham leads to a Hamming loss of 53% vs using click bait/written by humans leading to 19%
#
# Can you find better label descriptions?

# %%
data.columns

# %%
data.fillna(value={'headline':'', 'body':''}, inplace=True)

# %% tags=[]
data['text'] = data['headline']+" "+data['body']

# %%
data['text']=data.text.str.replace('\n',' ').str.replace('\r', ' ')


# %%
data.info()

# %%
data_1000 = data.query('ann_round==2').copy()

# %%
data_1000.shape

# %%
sequence = data_1000.iloc[66,:].text
sequence

# %%

candidate_labels = ["argumentation", 
                    "discriminating",
                    "inappropriate",
                    "a personal story", 
                    "off-topic",
                    "requiring feedback", 
                    "positive",
                    "neutral",
                    "negative"
                   ]
classifier(sequence, candidate_labels, multi_label=True)


# %%
data_1000.iloc[66,:]

# %%
category_map = {'label_argumentsused': "argumentation", 
                   'label_discriminating': "discriminating",
                    'label_inappropriate': "inappropriate",
                    'label_personalstories': "a personal story", 
                    'label_offtopic': "off-topic",
                    'label_possiblyfeedback': "requiring feedback", 
                    'label_sentimentpositive': "positive",
                    'label_sentimentneutral': "neutral",
                   'label_sentimentnegative': "negative"}
category_map = {value:key  for key, value in category_map.items() }

# %%
candidate_labels = list(category_map.keys())
predictedCategories = []
trueCategories = []
for i in tqdm(range(2)):
    text = data_1000.iloc[i,]['text']
    #cat = [data.iloc[i,]['target']]
    res = classifier(text, candidate_labels, multi_label=True)
    labels = res['labels'] 
    scores = res['scores'] #extracting the scores associated with the labels
    res_dict = {category_map[label] : score for label,score in zip(labels, scores)}
    res_dict['id_post'] = data_1000.iloc[i,]['id_post']
    predictedCategories.append(res_dict)

# %%
data_pred = pd.DataFrame(predictedCategories)

# %%
data_pred.to_csv('./output/zero_shot_result.csv')
