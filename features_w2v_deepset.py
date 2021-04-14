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
import numpy as np
import pandas as pd
import gensim
import codecs
import time
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import feature_engineering, loading


# %% [markdown]
# ## Read vectors

# %% [markdown]
# **TODO** Tidy up code: docstrings, variable names, ...

# %%
def load_embedding_model(file):
    """
    :param file: embeddings_path: path of file.
    :return: dictionary
    """

    with open(file, 'r') as f:
        model = {}
        for i, line in enumerate(f):
            split_line = line.split()
            word = ast.literal_eval(split_line[0]).decode('utf-8')
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
            #if i == 10000: break
    return model


# %%
start=time.time()
w2v_dict = load_embedding_model('./embeddings/w2v_vectors.txt')
end=time.time()
print(end-start)


# %% [markdown]
# ## MeanEmbedding

# %%
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or self.word2vec['UNK'], axis=0)
            for words in X
        ])


# %%
nb_pipeline = Pipeline([
        ('NBCV',MeanEmbeddingVectorizer(w2v_dict)),
        #('nb_norm', MinMaxScaler()),
        ('logreg',LogisticRegression())
    ])

# %%
