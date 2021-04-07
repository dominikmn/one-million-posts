# Modeling with Support Vector Machines #32
# Issue link: https://github.com/dominikmn/one-million-posts/issues/32

#import sys
#sys.path.append("..") # add project directory to system path

# general imports
import pandas as pd
import numpy as np
import re
from datetime import datetime

# NLP imports
from nltk.tokenize import word_tokenize
from nltk.stem.cistem import Cistem
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import GermanWortschatzLemmatizer as gwl  #https://docs.google.com/document/d/1rdn0hOnJNcOBWEZgipdDfSyjJdnv_sinuAUSDSpiQns/edit?hl=en#heading=h.oosx9e35prgf
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
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

## Optional preprocess fuction
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


## Params initialization
params = dict()

## Loading data
print("Loading data...")
#df_train = loading.load_extended_posts(split='train')
#df_test = loading.load_extended_posts(split='test')
#df_val = loading.load_extended_posts(split='val')
#
#df_train = feature_engineering.add_column_text(df_train)
#df_test = feature_engineering.add_column_text(df_test)
#df_val = feature_engineering.add_column_text(df_val)
#
#df_train.to_csv('./data/df_train.csv', sep='\t')
#df_test.to_csv('./data/df_test.csv', sep='\t')
#df_val.to_csv('./data/df_val.csv', sep='\t')

df_train = pd.read_csv('./data/df_train.csv', sep='\t')
df_test = pd.read_csv('./data/df_test.csv', sep='\t')
df_val = pd.read_csv('./data/df_val.csv', sep='\t')

df_combi = pd.concat([df_train,df_val])
fold = [-1]*df_train.shape[0] + [0]*df_val.shape[0]

print("Data loaded.")

y_col = ['label_argumentsused', 'label_discriminating', 'label_inappropriate', 'label_offtopic', 'label_personalstories',
            'label_possiblyfeedback', 'label_sentimentnegative', 'label_sentimentneutral', 'label_sentimentpositive']

def train_model(label,):
    print('-'*50)
    print(f"Model building for label: {label}.")
    params['label'] = label

    X_train = df_train.text
    X_test = df_test.text
    X_val = df_val.text
    X_combi = df_combi.text
    y_train = df_train[params['label']]
    y_test = df_test[params['label']]
    y_val = df_val[params['label']]
    y_combi = df_combi[params['label']]

    ## Feature extraction
    params["normalization"] = 'lower'
    #params["stopwords"] = None
    vectorizer = CountVectorizer(stop_words=None, preprocessor=None, tokenizer=None)
    X_train_vector = vectorizer.fit_transform(X_train)
    X_val_vector = vectorizer.transform(X_val)
    X_test_vector = vectorizer.transform(X_test)
    X_combi_vector = vectorizer.transform(X_combi)
    params["vectorizer"] = ''.join(e for e in str(vectorizer) if e.isalnum())

    # setting the MLFlow connection and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))

    #### Modeling
    #params["regularization_c"] =  0.1
    #model = SVC(C=params['regularization_c'], kernel='linear')
    model = SVC()
    pipeline = Pipeline([('model', model)])

    ps = PredefinedSplit(fold)
    params_grid = {'model__C':[1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6],
        'model__kernel':['linear', 'rbf']
        }
    grid_search = GridSearchCV(pipeline, param_grid=params_grid, cv=ps, scoring='f1',
                           verbose=5, n_jobs=-1)

    #print("Starting fit...")
    grid_search.fit(X_combi_vector, y_combi)

    # Predict
    model_best = grid_search.best_estimator_['model']
    params['model'] = ''.join(e for e in str(model_best) if e.isalnum())
    y_train_pred = model_best.predict(X_train_vector)
    y_val_pred = model_best.predict(X_val_vector)
    y_test_pred = model_best.predict(X_test_vector)

    #### MlFlow logging
    # Logging params, metrics, and model with mlflow:
    mlflow.log_metric("train - " + "F1", f1_score(y_train, y_train_pred))
    mlflow.log_metric("train - " + "recall", recall_score(y_train, y_train_pred))
    mlflow.log_metric("train - " + "precision", precision_score(y_train, y_train_pred))

    mlflow.log_metric("val - " + "F1", f1_score(y_val, y_val_pred))
    mlflow.log_metric("val - " + "recall", recall_score(y_val, y_val_pred))
    mlflow.log_metric("val - " + "precision", precision_score(y_val, y_val_pred))

    ##mlflow.log_metric("test - " + "F1", f1_score(y_test, y_test_pred))
    ##mlflow.log_metric("test - " + "recall", recall_score(y_test, y_test_pred))
    ##mlflow.log_metric("test - " + "precision", precision_score(y_test, y_test_pred))

    mlflow.log_params(params)
    mlflow.set_tag("running_from_jupyter", "False")
    #mlflow.log_artifact("./models")
    #mlflow.sklearn.log_model(model, "model")
    path = f"models/{params['model']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    save_model(sk_model=model_best, path=path)

    mlflow.end_run()


if __name__ == '__main__':
    for l in y_col:
        train_model(l) 