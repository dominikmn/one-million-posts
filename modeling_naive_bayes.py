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
#import sys
#sys.path.append("..") # add project directory to system path

# general imports
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer

# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# evaluation imports
from sklearn.metrics import f1_score, recall_score, precision_score

# project utils
from utils import loading, feature_engineering

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


# %%
def __get_data():
    df_posts_train = loading.load_extended_posts(split="train")
    df_posts_val = loading.load_extended_posts(split="val")

    df_train = feature_engineering.add_column_text(df_posts_train)
    df_val = feature_engineering.add_column_text(df_posts_val)

    target_labels = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
           'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
           'label_sentimentnegative', 'label_sentimentneutral', 'label_sentimentpositive',]
    X_train = df_train.text
    y_train = df_train[target_labels]
    X_val = df_val.text
    y_val = df_val[target_labels]
    return X_train, X_val, y_train, y_val


# %% [markdown]
# ## Feature extraction


# %% [markdown]
# ## Modeling
#

# %%
def __compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, split: str="train"
) -> (float, float, float):
    """Computes and logs metrics to mlflow and logger

    Args:
        y_true: The true target classification
        y_pred: The predicted target classification
        split: The split of the dataset ["test", "val", "train"]

    Returns:
        f1: The f1_score
        precision: The precision_score
        recall: The recall_score
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logger.info(
        f"Performance on "
        + str(split)
        + f" set: F1 = {f1:.1f}, precision = {precision:.1%}, recall = {recall:.1%}"
    )
    mlflow.log_metric(f"{split} - F1", f1)
    mlflow.log_metric(f"{split} - recall", recall)
    mlflow.log_metric(f"{split} - precision", precision)

    return f1, precision, recall


# %%
def run_training(model_details, mlflow_params):
    logger.info(f"Getting the data")
    X_train, X_val, y_train_multi, y_val_multi = __get_data()

    logger.info("Training simple model and tracking with MLFlow")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # model
    logger.info(f"Training a simple {model_details['name']}")

    # label
    for label in y_train_multi.columns:
        y_train = y_train_multi[label]
        y_val = y_val_multi[label]
        mlflow_params["label"] = label

        logger.info(f"Training a simple {model_details['name']} for {label}")
        with mlflow.start_run():
            model = model_details["model"].fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            if isinstance(model, GridSearchCV):
                best_params = model.best_params_
                if 'vectorizer__stop_words' in best_params.keys() and best_params['vectorizer__stop_words']!=None:
                    best_params['vectorizer__stop_words'] = "NLTK-German"
                mlflow_params["best_params"] = best_params

            mlflow.log_params(mlflow_params)
            __compute_and_log_metrics(y_train, y_train_pred, "train")
            __compute_and_log_metrics(y_val, y_val_pred, "val")

            # saving the model
            logger.info("Saving model in the models folder")
            t = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            path = f"models/{model_details['name']}_{label}_{t}"
            save_model(sk_model=model, path=path)


# %%
if __name__ == "__main__":
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words=stopwords)),
        ("clf", MultinomialNB()),
    ])
    param_grid = {
        "vectorizer__ngram_range": [(1,1),],
        "vectorizer__min_df": np.linspace(0, 0.05, 2),
        "vectorizer__max_df": np.linspace(0.95, 1.0, 2),
    }
    gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=3, verbose=3)

    model = {"name": "NaiveBayes", "model": gs}
    mlflow_params = {
        "vectorizer": "tfidf",
        "normalization": "lower",
        "stopwords": "nltk-german",
        "model": model["name"],
        "grid_search_params": param_grid,
    }

    run_training(model, mlflow_params)

# %%
