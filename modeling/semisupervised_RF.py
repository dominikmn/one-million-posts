import numpy as np

# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from utils import modeling as m
from utils import transformers

import mlflow
from modeling.config import TRACKING_URI, EXPERIMENT_NAME#, TRACKING_URI_DEV
import logging

# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    TARGET_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
        'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
        'label_sentimentnegative', 'label_sentimentpositive',]

    LABEL = 'label_sentimentnegative'
    WORD2VEC = transformers.load_embedding_vectors(embedding_style='word2vec', file="./embeddings/word2vec_vectors.txt")
    BALANCE_METHOD = "translate"
    SAMPLING_STRATEGY = 0.9
    RUN_NAME = "semisupervised_RF_first_try"

    rf = RandomForestClassifier(random_state=42, min_samples_leaf=5, max_depth=20)
    pipeline = Pipeline([
        ("vectorizer", transformers.MeanEmbeddingVectorizer(embedding_dict=WORD2VEC)),
        ("clf", SelfTrainingClassifier(rf, verbose=1, criterion='k_best',
                                             k_best=100, max_iter=None)),
    ])

    mlflow_params = {
        "vectorizer": "word2vec",
        "model": "SemSup_RandomForest",
        "clf__k_best": 100,
        "balance_method": BALANCE_METHOD,
        "sampling_strategy": SAMPLING_STRATEGY,
    }
    mlflow_tags = {
        "cycle6": True,
    }

    IS_DEVELOPMENT = False

    data = m.Posts()
    data.set_semi_supervised(True)
    mlflow_logger = m.MLFlowLogger(
        uri=TRACKING_URI,
        experiment=EXPERIMENT_NAME,
        is_dev=IS_DEVELOPMENT,
        params=mlflow_params,
        tags=mlflow_tags
    )
    training = m.Modeling(data, pipeline, mlflow_logger)
    for label in LABEL:
        logger.info(f"-"*20)
        logger.info(f"Target: {label}")
        data.set_label(label=label)
        data.set_balance_method(balance_method=BALANCE_METHOD, sampling_strategy=SAMPLING_STRATEGY)
        training.train()
        training.evaluate(["train", "val"])
        with mlflow.start_run(run_name=RUN_NAME) as run:
            mlflow_logger.log()
