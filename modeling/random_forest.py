# -*- coding: utf-8 -*-
import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')


# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler

from utils import modeling as m
from utils import cleaning, transformers

import mlflow
from modeling.config import TRACKING_URI, EXPERIMENT_NAME#, TRACKING_URI_DEV
import logging

from time import time

# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    
    data = m.Posts()

    trans_os = {'translate':[1.0], 'oversample':[1.0]}

    TARGET_LABELS = ['label_discriminating', 'label_inappropriate', 'label_offtopic',
        'label_sentimentnegative', 'label_needsmoderation', 'label_negative']

    embedding_dict_glove = transformers.load_embedding_vectors(embedding_style='glove', file="./embeddings/glove_vectors.txt")
    embedding_dict_word2vec = transformers.load_embedding_vectors(embedding_style='word2vec', file="./embeddings/w2v_vectors.txt")
    
    preps = {
            'norm': lambda x: cleaning.series_apply_chaining(x, [cleaning.normalize]),
            'glove': transformers.MeanEmbeddingVectorizer(embedding_dict=embedding_dict_glove).transform,
            'word2vec': transformers.MeanEmbeddingVectorizer(embedding_dict=embedding_dict_word2vec).transform,
            }
    vecs = {
            'count': CountVectorizer(),
           'tfidf': TfidfVectorizer(),
            }
    PRE_VEC_COMBINATIONS = [
                        ['glove', 'glove'],
                        ['word2vec', 'word2vec'],
                        ['norm', 'count'],
                        ['norm', 'tfidf'],
                    ]
    
    for method, strat in trans_os.items():
        for strategy in strat:
            print(method, strategy)
            for label in TARGET_LABELS:
                for c in PRE_VEC_COMBINATIONS:
                    mlflow_params=dict()
                    print(c)
                    constant_preprocessor=preps[c[0]]
                    if c[1] in ['count', 'tfidf']:
                        pipeline = Pipeline([
                            ("vectorizer", vecs[c[1]]),
                            ("clf", RandomForestClassifier(random_state=42)),
                            ])
                        param_grid = {
                            "vectorizer__ngram_range" : [(1,1), (1,2), (1,3)],
                            "vectorizer__stop_words" : [stopwords, None],
                            "vectorizer__min_df": [0.],
                            "vectorizer__max_df": [0.9],
                            'clf__max_depth': [5, 10, 40],
                            'clf__min_samples_leaf': [1, 5, 10],
                        }
                        grid_search_params = param_grid.copy()
                        # MLFlow params have limited characters, therefore stopwords must not be given as list
                        grid_search_params["vectorizer__stop_words"] = ["NLTK-German", None]
                        mlflow_params["normalization"] = c[0]
                        mlflow_params["vectorizer"]    = c[1]
                    else:
                        pipeline = Pipeline([
                            ("clf", RandomForestClassifier(random_state=42)),
                            ])
                        param_grid = {
                            'clf__max_depth': [5, 10, 40],
                            'clf__min_samples_leaf': [1, 5, 10],
                        }
                        grid_search_params = param_grid.copy()
                        mlflow_params["normalization"] = 'norm'
                        mlflow_params["vectorizer"]    = c[1]

                    gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, verbose=1, n_jobs=-1)

                    mlflow_params["model"]=  "RandomForest"
                    mlflow_params["grid_search_params"]=  str(grid_search_params)[:249]
                    mlflow_tags = {
                        "cycle3": True,
                    }

                    IS_DEVELOPMENT = False

                    mlflow_logger = m.MLFlowLogger(
                        uri=TRACKING_URI,
                        experiment=EXPERIMENT_NAME,
                        is_dev=IS_DEVELOPMENT,
                        params=mlflow_params,
                        tags=mlflow_tags
                    )
                    training = m.Modeling(data, gs, mlflow_logger)

                    logger.info(f"-"*20)
                    logger.info(f"Target: {label}")
                    data.set_label(label=label)
                    data.set_balance_method(balance_method=method, sampling_strategy=strategy)
                    training.train(constant_preprocessor=constant_preprocessor)
                    training.evaluate(["train", "val"],constant_preprocessor=constant_preprocessor)
                    #if True:
                    with mlflow.start_run(run_name='randomforest_with_full_params') as run:
                        mlflow_logger.log()



