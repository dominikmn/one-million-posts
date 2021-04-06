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

# %%

# %%
#import sys
#sys.path.append("..") # add project directory to system path

# general imports
import pandas as pd
import numpy as np
import re

# NLP imports
from nltk.tokenize import word_tokenize
from nltk.stem.cistem import Cistem
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import GermanWortschatzLemmatizer as gwl  #https://docs.google.com/document/d/1rdn0hOnJNcOBWEZgipdDfSyjJdnv_sinuAUSDSpiQns/edit?hl=en#heading=h.oosx9e35prgf
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer

# modeling imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# evaluation imports
from sklearn.metrics import f1_score, recall_score, precision_score

# project utils
from utils import loading, feature_engineering

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME

# %%
df_posts = loading.load_extended_posts()
df_posts = feature_engineering.add_column_ann_round(df_posts)

# %% [markdown]
# ## Data selection
#
# Choose only posts from the second annotation round to reduce the sample size.
#
# The first simple model will only use the posts' text for classification. Since headline and body build the text, we will simply concatenate them to get the corpus.

# %%
df = df_posts.query("ann_round == 2")

# %% [markdown]
# # TODO make function

# %%
df = df.fillna(value={"body": "", "headline": ""})
df["text"] = df.body + df.headline
df.text = df.text.str.replace("\n", " ").str.replace("\r", " ")

# %%
df.head()

# %% [markdown]
# ### Train-val-test split
#
# Thoughts concerning the split:
#
# * two test sets, one sampled from annotation round 2 and the second one from round 3 (due to the sampling stategies of the authors)
# * should we include article category/topic in stratification
# * train-val-test split
# * how did the authors split? 10-fold CrossValidation, 1 fold as test set for each "round"; resulting scores (stated in paper 1) are therefore CV-test-scores

# %%
df.columns

# %%
X = df.text
y_col = ['label_argumentsused', 'label_discriminating', 'label_inappropriate', 'label_offtopic', 'label_personalstories',
         'label_possiblyfeedback', 'label_sentimentnegative', 'label_sentimentneutral', 'label_sentimentpositive']
y = df.label_sentimentnegative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24, stratify=y)


# %% [markdown]
# ## Feature extraction

# %%
def preprocess_text(text):
    # normalize (remove capitalization and punctuation)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize
    words = text.split()
    return words

    # stopwords removal
    words = [w for w in words if w not in stopwords]

    # stemming
    stemmer = Cistem()
    stemmed = [stemmer.stem(w) for w in words]

    # lemmatization
    # TODO: find/install german lemmatizer
    
    return stemmed


# %%
vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train_trans = vectorizer.fit_transform(X_train)

# %% [markdown]
# ## Modeling
#

# %% [markdown]
# #### MLFlow Setup

# %%
# !ps -A | grep gunicorn

# %%
TRACKING_URI

# %%
EXPERIMENT_NAME

# %%
# setting the MLFlow connection and experiment
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# %%
mlflow.start_run()
run = mlflow.active_run()
print("Active run_id: {}".format(run.info.run_id))

# %% [markdown]
# #### Modeling

# %%
# Build and fit the model
logreg = LogisticRegression()
model = logreg.fit(X_train_trans, y_train)

# %% [markdown]
# Predict on **training data**.

# %%
y_train_pred = model.predict(X_train_trans)

# %%
mlflow.log_metric("train -" + "F1", f1_score(y_train, y_train_pred))
mlflow.log_metric("train -" + "recall", recall_score(y_train, y_train_pred))
mlflow.log_metric("train -" + "precision", precision_score(y_train, y_train_pred))

# %% [markdown]
# Predict on **test data**.

# %%
X_test_trans = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_trans)

# %%
mlflow.log_metric("test -" + "F1", f1_score(y_test, y_test_pred))
mlflow.log_metric("test -" + "recall", recall_score(y_test, y_test_pred))
mlflow.log_metric("test -" + "precision", precision_score(y_test, y_test_pred))

# %% [markdown]
# #### MlFlow logging

# %% [markdown]
# Logging params, metrics, and model with mlflow:

# %%
params = {
    "vectorizer": "tfidf",
    "normalization": "lower"
}

# %%
mlflow.log_params(params)
mlflow.set_tag("running_from_jupyter", "True")
#mlflow.log_artifact("./models")
#mlflow.sklearn.log_model(model, "model")
path = "models/linear"
save_model(sk_model=model, path=path)
mlflow.end_run()

# %%
