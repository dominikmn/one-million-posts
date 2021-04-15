
import pandas as pd
import numpy as np
from utils import loading, feature_engineering, scoring
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
SEED=42
from datetime import datetime

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME


# -

def get_data():
    data_train = loading.load_extended_posts(split='train')
    data_train = feature_engineering.add_column_ann_round(data_train)
    data_val = loading.load_extended_posts(split='val')
    data_val = feature_engineering.add_column_ann_round(data_val)
    return data_train, data_val


def clean_data(df):
    df.fillna(value={'headline':'', 'body':''}, inplace=True)
    df['text'] = df['headline']+" "+df['body']
    df['text']=df.text.str.replace('\n',' ').str.replace('\r', ' ')
    return df


def split_Xy(df, y_cols):
    X = df.text
    y = df[y_cols]
    return X, y



def get_pipe_grid():
    pipe = Pipeline(steps = [('vectorizer', TfidfVectorizer()), ('classifier',RandomForestClassifier(random_state=SEED,class_weight='balanced_subsample'))])
    grid = {
    'vectorizer__ngram_range' : [(1,1), (1,2), (1,3)],
    'vectorizer__stop_words' : [stopwords, None],
    'vectorizer__max_df': [0.5, 1.0],
    'vectorizer__min_df': [0, 5],
    'classifier__n_estimators': [100, 1000], 
    'classifier__max_depth': [1, 5, 10],
    'classifier__min_samples_leaf': [5, 10, 40]
    }
    return pipe, grid



def start_mlflow():
    # !ps -A | grep gunicorn
    # setting the MLFlow connection and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    return run


def cv_rf(X_train, y_train, X_val, pipe, grid, label):
    search = GridSearchCV(estimator = pipe,
                    param_grid = grid,
                    scoring = 'f1',
                    cv = 5,
                    n_jobs = -1,
                    verbose = 2)
    search.fit(X_train, y_train[label])
    model = search.best_estimator_
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)
    y_pred_val_proba = model.predict_proba(X_val)[:,1]
    y_pred_train_proba = model.predict_proba(X_train)[:,1]
    best_params=search.best_params_
    if best_params['vectorizer__stop_words']!=None:
        best_params['vectorizer__stop_words']='NLTK-German'
    return model,best_params, pd.Series(y_pred_val, name=label), pd.Series(y_pred_train, name=label), pd.Series(y_pred_val_proba, name=label), pd.Series(y_pred_train_proba, name=label)


def best_th(y_pred_proba, y_true, label):
        best_th = 0.0
        best_f1 = 0.0
        for th in np.arange(0.05, 0.96, 0.05):
            y_pred_temp = (y_pred_proba >= th).astype(int)
            if f1_score(y_true[label], y_pred_temp)>best_f1:
                best_th = th
                best_f1 = f1_score(y_true[label], y_pred_temp)
        return best_th


def log_metrics(f1_val, recall_val, precision_val, f1_train, recall_train, precision_train):
    mlflow.log_metric("val - " + "F1", f1_val)
    mlflow.log_metric("val - " + "recall", recall_val)
    mlflow.log_metric("val - " + "precision", precision_val)
    mlflow.log_metric("train - " + "F1", f1_train)
    mlflow.log_metric("train - " + "recall", recall_train)
    mlflow.log_metric("train - " + "precision", precision_train)


def sv_model(params, model):
    mlflow.log_params(params)
    mlflow.set_tag("running_from_jupyter", "True")
    #mlflow.log_artifact("./models")
    #mlflow.sklearn.log_model(model, "model")
    path = f"models/{params['model']+params['label']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    save_model(sk_model=model, path=path)


def endrun():
    mlflow.end_run()

def get_params(label, best_params):
    params = {
    "model": "random_forest_cv_testrun",
    "label": label,
    "vectorizer": "tfidf",
    "normalization": "lower",
    "best_params": best_params
    }
    return params
    
# +
def train_rf():
    data_train, data_val = get_data()
    data_train=clean_data(data_train)
    data_val=clean_data(data_val)
    
    y_cols=['label_argumentsused', 'label_discriminating', 'label_inappropriate',
       'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
       'label_sentimentnegative',
        'label_sentimentneutral',
       'label_sentimentpositive']
    

    
    pipe, grid = get_pipe_grid()
    grid_mlflow = {
    'vec_ngram_range' : [(1,1), (1,2), (1,3)],
    'vec_stop_words' : ['NLTK-German', None],
    'vec_max_df': [0.5, 1.0],
    'vec_min_df': [0, 5],
    'cl_n_estimators': [100, 1000], 
    'cl_max_depth': [1, 5, 10],
    'cl_min_samples_leaf': [5, 10, 40]
    }
    
    
    for label in y_cols:
        X_train, y_train = split_Xy(data_train.dropna(subset = [label]), y_cols)
        X_val, y_val = split_Xy(data_val, y_cols)
        endrun()
        start_mlflow()
        model,best_params, y_pred_val, y_pred_train, y_pred_val_proba, y_pred_train_proba = cv_rf(X_train, y_train, X_val, pipe, grid, label)
        params = get_params(label, best_params)
        best_thr = best_th(y_pred_train_proba, y_train, label)
        params['threshold']=best_thr
        params['grid_search_params']=grid_mlflow
        y_pred_th_val = (y_pred_val_proba >= best_thr).astype(int)
        y_pred_th_train = (y_pred_train_proba >= best_thr).astype(int)
        f1_val = f1_score(y_val[label], y_pred_th_val)
        precision_val = precision_score(y_val[label], y_pred_th_val)
        recall_val = recall_score(y_val[label], y_pred_th_val)
        f1_train = f1_score(y_train[label], y_pred_th_train)
        precision_train = precision_score(y_train[label], y_pred_th_train)
        recall_train = recall_score(y_train[label], y_pred_th_train)
        scoring.log_cm(y_train[label], y_pred_th_train, y_val[label], y_pred_th_val)
        log_metrics(f1_val, recall_val, precision_val, f1_train, recall_train, precision_train)
        sv_model(params, model)
        print(f'{label} f1-score(validation): {f1_val}')
        endrun()
        
    
    
    
# -
if __name__ == "__main__":
    train_rf()
