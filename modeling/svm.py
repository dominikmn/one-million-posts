# Modeling with Support Vector Machines #32
# Issue link: https://github.com/dominikmn/one-million-posts/issues/32

#import sys
#sys.path.append("..") # add project directory to system path

# general imports
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging

# NLP imports
#from nltk.tokenize import word_tokenize
#from nltk.stem.cistem import Cistem
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import GermanWortschatzLemmatizer as gwl  #https://docs.google.com/document/d/1rdn0hOnJNcOBWEZgipdDfSyjJdnv_sinuAUSDSpiQns/edit?hl=en#heading=h.oosx9e35prgf
#from nltk.corpus import stopwords
#stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# modeling imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score

# project utils
from utils import loading, feature_engineering

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)

# Optional preprocess fuction
def preprocess_snowball(text):
    # normalize (remove capitalization and punctuation)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    words = text.split()

    # stemming
    stemmer = SnowballStemmer(language='german')
    stemmed = [stemmer.stem(w) for w in words]
    
    return stemmed

# Params initialization
mlflow_params = dict()

# Loading data
logger.info("Loading data...")
df_train = loading.load_extended_posts(split='train')
df_test = loading.load_extended_posts(split='test')
df_val = loading.load_extended_posts(split='val')

df_train = feature_engineering.add_column_text(df_train)
df_test = feature_engineering.add_column_text(df_test)
df_val = feature_engineering.add_column_text(df_val)

df_train.to_csv('./data/df_train.csv', sep='\t')
df_test.to_csv('./data/df_test.csv', sep='\t')
df_val.to_csv('./data/df_val.csv', sep='\t')

df_train = pd.read_csv('./data/df_train.csv', sep='\t')
df_test = pd.read_csv('./data/df_test.csv', sep='\t')
df_val = pd.read_csv('./data/df_val.csv', sep='\t')

logger.info("Data loaded.")

y_col = [
        'label_sentimentnegative', 
        'label_sentimentpositive',
        'label_offtopic', 
        'label_inappropriate', 
        'label_discriminating', 
        'label_possiblyfeedback', 
        'label_personalstories', 
        'label_argumentsused', 
        #'label_sentimentneutral', 
        ]

def train_model(label,):
    print('-'*50)
    logger.info(f"Model-building for label: {label}.")

    X_train = df_train.text
    X_val = df_val.text
    X_test = df_test.text

    y_train = df_train[label]
    y_val = df_val[label]
    y_test = df_test[label]

    # setting the MLFlow connection and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        run = mlflow.active_run()
        logger.info("Active run_id: {}".format(run.info.run_id))

        #### Modeling
        vectorizer = CountVectorizer()
        model = SVC()
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('model', model),
            ])

        grid_search_params = {
            #'vectorizer__stop_words': [None], 
            #'vectorizer__preprocessor': [None, preprocess_snowball],
            #'vectorizer__tokenizer': [None],
            'model__C':[.5**0, .5**1, .5**2, .5**3, .5**4, .5**5,],
            'model__kernel':['linear', 'rbf',],
            }
        grid_search = GridSearchCV(pipeline, param_grid=grid_search_params, cv=3, scoring='f1',
                            verbose=3, n_jobs=-1)

        logger.info("Start fitting...")
        grid_search.fit(X_train, y_train)

        # Predict
        logger.info(f"Best score: {grid_search.best_score_}")
        model_best = grid_search.best_estimator_
        y_train_pred = model_best.predict(X_train)
        y_val_pred = model_best.predict(X_val)
        y_test_pred = model_best.predict(X_test)

        #### MlFlow logging
        # Logging params, metrics, and model with mlflow:
        #mlflow.log_metric("train - " + "F1", f1_score(y_train, y_train_pred))
        #mlflow.log_metric("train - " + "recall", recall_score(y_train, y_train_pred))
        #mlflow.log_metric("train - " + "precision", precision_score(y_train, y_train_pred))

        #mlflow.log_metric("val - " + "F1", f1_score(y_val, y_val_pred))
        #mlflow.log_metric("val - " + "recall", recall_score(y_val, y_val_pred))
        #mlflow.log_metric("val - " + "precision", precision_score(y_val, y_val_pred))

        #mlflow.log_metric("test - " + "F1", f1_score(y_test, y_test_pred))
        #mlflow.log_metric("test - " + "recall", recall_score(y_test, y_test_pred))
        #mlflow.log_metric("test - " + "precision", precision_score(y_test, y_test_pred))

        mlflow_params["normalization"] = 'lower'
        mlflow_params["vectorizer"] = ''.join(e for e in str(vectorizer) if e.isalnum())
        mlflow_params['model'] = ''.join(e for e in str(model) if e.isalnum())
        mlflow_params['grid_search_params'] = grid_search_params
        mlflow_params['label'] = label
        mlflow_params["best_params"] = grid_search.best_params_
        #mlflow.log_params(mlflow_params)
        #mlflow.set_tag("running_from_jupyter", "False")
        #mlflow.log_artifact("./models")
        #mlflow.sklearn.log_model(model, "model")
        t = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = f"models/{mlflow_params['model']}_{label}_{t}"
        #save_model(sk_model=model_best, path=path)

SCORES_BOW = {
        'label_sentimentnegative':{'f1':0.5307},
        'label_sentimentpositive':{'f1':0.0822},
        'label_offtopic':{'f1':0.2553}, 
        'label_inappropriate':{'f1':0.1328}, 
        'label_discriminating':{'f1':0.1321}, 
        'label_possiblyfeedback':{'f1':0.6156}, 
        'label_personalstories':{'f1':0.6407}, 
        'label_argumentsused':{'f1':0.5625},
        }

SCORES_2017 = {
        'BOW': SCORES_BOW,
        }

def train_pseudo(label, model):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
        
    #mlflow.log_metric("train - " + "F1", )
    #mlflow.log_metric("train - " + "recall", recall_score(y_train, y_train_pred))
    #mlflow.log_metric("train - " + "precision", precision_score(y_train, y_train_pred))
    mlflow.log_metric("val - " + "F1", SCORES_2017[model][label]['f1'] )
    #mlflow.log_metric("val - " + "recall", recall_score(y_val, y_val_pred))
    #mlflow.log_metric("val - " + "precision", precision_score(y_val, y_val_pred))
    #mlflow.log_metric("test - " + "F1", f1_score(y_test, y_test_pred))
    #mlflow.log_metric("test - " + "recall", recall_score(y_test, y_test_pred))
    #mlflow.log_metric("test - " + "precision", precision_score(y_test, y_test_pred))

    mlflow_params['label'] = label
    mlflow_params['model'] = model
    mlflow_params["normalization"] = ''
    mlflow_params["vectorizer"] = ''
    mlflow_params['grid_search_params'] = ''
    mlflow_params["best_params"] = ''
    mlflow.log_params(mlflow_params)
    mlflow.end_run()


if __name__ == '__main__':
    for l in y_col:
        #train_pseudo(l, 'BOW')
        train_model(l) 