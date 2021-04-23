import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
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
        ("clf", DummyClassifier(strategy='constant', constant=1)),
    ])
   

    TARGET_LABELS = ['label_discriminating', 'label_inappropriate',
        'label_sentimentnegative', 'label_needsmoderation',]
    mlflow_params=dict()
    mlflow_params["model"]=  "DummyClassifier"
    mlflow_tags = {"cycle4": True,}
    
    IS_DEVELOPMENT = False

    data = m.Posts()
    mlflow_logger = m.MLFlowLogger(
        uri=TRACKING_URI,
        experiment=EXPERIMENT_NAME,
        is_dev=IS_DEVELOPMENT,
        params=mlflow_params,
        tags=mlflow_tags
    )
    training = m.Modeling(data, pipeline, mlflow_logger)
    for label in TARGET_LABELS:
        logger.info(f"-"*20)
        logger.info(f"Target: {label}")
        data.set_label(label=label)
        data.set_balance_method(balance_method=None, sampling_strategy=None)
        training.train()
        training.evaluate(["train", "val"])
        #if True:
        with mlflow.start_run(run_name='dummy_classifier') as run:
            mlflow_logger.log()
    