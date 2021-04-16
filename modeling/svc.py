# -*- coding: utf-8 -*-
import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')


# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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
    #embedding_dict_glove = transformers.load_embedding_vectors(embedding_style='glove')
    #embedding_dict_w2v = transformers.load_embedding_vectors(embedding_style='word2vec')

    trans_os = {None: [None], 'translate':[0.8,0.9,1.0], 'oversample':[0.8,0.9,1.0]} 
    
    #vecs = {CountVectorizer(): 'count', 
    #        TfidfVectorizer(): 'tfidf',
    #        transformers.MeanEmbeddingVectorizer(embedding_dict=embedding_dict_glove): 'glove',
    #       transformers.MeanEmbeddingVectorizer(embedding_dict=embedding_dict_w2v): 'word2vec',
    #       }
    vecs = {CountVectorizer(): 'count',
           TfidfVectorizer(): 'tfidf',
           }
    
    lem = cleaning.lem_germ
    stem = cleaning.stem_germ
    norm = cleaning.normalize
    
    
    

    for vec, vec_name in vecs.items():
        print(vec_name)

        if vec_name in ['count', 'tfidf']:
            pipeline = Pipeline([
                ("vectorizer", vec),
                ("clf", SVC()),
                ])
            param_grid = {
                "vectorizer__ngram_range" : [(1,1), (1,2), (1,3)],
                "vectorizer__stop_words" : [stopwords, None],
                "vectorizer__min_df": np.linspace(0, 0.1, 3),
                "vectorizer__max_df": np.linspace(0.9, 1.0, 3),
                "vectorizer__preprocessor": [norm, stem, lem],
                "clf__C": [.5**0, .5**1, .5**2, .5**3, .5**4, .5**5],
                "clf__kernel": ['linear', 'rbf'],
            }
            
        else:
            pipeline = Pipeline([
                ("vectorizer", vec),
                ("clf", SVC()),
                ])

            param_grid = {
                "clf__C": [.5**0, .5**1, .5**2, .5**3, .5**4, .5**5],
                "clf__kernel": ['linear', 'rbf'],
            }
        # For clear logging output use verbose=1
        gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, verbose=1)

        # MLFlow params have limited characters, therefore stopwords must not be given as list
        grid_search_params = param_grid.copy()
        grid_search_params["vectorizer__stop_words"] = ["NLTK-German", None]
        if vec_name in ['count', 'tfidf']:
            grid_search_params["vectorizer__preprocessor"] = ["norm", "lem", "stem"]

        mlflow_params = {
            "vectorizer": vec_name,
            "normalization": "lower",
            "model": "SVM",
            "grid_search_params": str(grid_search_params)[:249],
        }
        mlflow_tags = {
            "cycle2": True,
        }

        TARGET_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
            'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
            'label_sentimentnegative', 'label_sentimentpositive',]

        IS_DEVELOPMENT = False


        mlflow_logger = m.MLFlowLogger(
            uri=TRACKING_URI,
            experiment=EXPERIMENT_NAME,
            is_dev=IS_DEVELOPMENT,
            params=mlflow_params,
            tags=mlflow_tags
        )
        training = m.Modeling(data, gs, mlflow_logger)
        for method, strat in trans_os.items():
            for strategy in strat:
                print(method, strategy)
                for label in TARGET_LABELS:
                    logger.info(f"-"*20)
                    logger.info(f"Target: {label}")
                    data.set_label(label=label)
                    data.set_balance_method(balance_method=method, sampling_strategy=strategy)
                    training.train()
                    training.evaluate(["train", "val"])
                    #if True:
                    with mlflow.start_run() as run:
                        mlflow_logger.log()



