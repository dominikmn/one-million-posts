import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import modeling as m

import mlflow
from modeling.config import TRACKING_URI, EXPERIMENT_NAME#, TRACKING_URI_DEV
import logging

# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", MultinomialNB()),
    ])
    param_grid = {
        "vectorizer__ngram_range" : [(1,1), (1,2), (1,3)],
        "vectorizer__stop_words" : [stopwords, None],
        "vectorizer__min_df": np.linspace(0, 0.1, 3),
        "vectorizer__max_df": np.linspace(0.9, 1.0, 3),
    }
    # For clear logging output use verbose=1
    gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=3, verbose=1)

    # MLFlow params have limited characters, therefore stopwords must not be given as list
    grid_search_params = param_grid.copy()
    grid_search_params["vectorizer__stop_words"] = ["NLTK-German", None]

    mlflow_params = {
        "vectorizer": "count",
        "normalization": "lower",
        "model": "NaiveBayes",
        "grid_search_params": grid_search_params,
    }
    mlflow_tags = {
        "cycle2": True,
    }

    TARGET_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
        'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
        'label_sentimentnegative', 'label_sentimentpositive',]
    
    IS_DEVELOPMENT = True

    data = m.Posts()
    mlflow_logger = m.MLFlowLogger(
        uri=TRACKING_URI,
        experiment=EXPERIMENT_NAME,
        is_dev=IS_DEVELOPMENT,
        params=mlflow_params,
        tags=mlflow_tags
    )
    training = m.Modeling(data, pipeline, mlflow_logger)
    for label in TARGET_LABELS[:1]:
        logger.info(f"-"*20)
        logger.info(f"Target: {label}")
        data.set_label(label=label)
        data.set_balance_method(balance_method=None, sampling_strategy=None)
        training.train()
        training.evaluate(["train", "val"])
        if True:
        #with mlflow.start_run() as run:
            mlflow_logger.log()
    