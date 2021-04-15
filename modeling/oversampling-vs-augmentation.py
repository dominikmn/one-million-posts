from utils import loading, scoring, augmenting
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
import numpy as np
import pandas as pd
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME
from datetime import datetime


def start_mlflow():
    # # !ps -A | grep gunicorn
    # setting the MLFlow connection and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    return run


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


    
if __name__=='__main__':
    df_train = loading.load_extended_posts(split = 'train')
    df_val = loading.load_extended_posts(split = 'val')

    df_train.fillna(value={'headline':'', 'body':''}, inplace=True)
    df_train['text'] = df_train['headline']+" "+df_train['body']
    df_train['text']=df_train.text.str.replace('\n',' ').str.replace('\r', ' ')
    df_val.fillna(value={'headline':'', 'body':''}, inplace=True)
    df_val['text'] = df_val['headline']+" "+df_val['body']
    df_val['text']=df_val.text.str.replace('\n',' ').str.replace('\r', ' ')

    y_cols = ['label_argumentsused', 'label_discriminating', 'label_inappropriate', 'label_offtopic', 'label_personalstories', 'label_possiblyfeedback', 'label_sentimentnegative', 'label_sentimentpositive']

    for label in y_cols:
        endrun()
        start_mlflow()
        params = params = {
        "model": "NB_oversampling_vs_augmenting",
        "label": label,
        "vectorizer": "tfidf",
        "normalization": "lower",
        "aug_or_os": "os"}
        best_f1_val=0
        best_perc=0


        for perc in np.arange(0.4, 1.01, 0.1):
            try:
                df_temp = df_train.dropna(subset=[label]).copy()
                X_train= np.array(df_temp['text']).reshape(-1, 1)
                y_train = np.array(df_temp[label]).reshape(-1, 1)
                X_val = df_val['text']
                y_val=  df_val[label]
                os = RandomOverSampler(sampling_strategy=perc, random_state=42)
                X_over, y_over = os.fit_resample(X_train, y_train)
                vec = TfidfVectorizer(stop_words=stopwords)
                X_over = vec.fit_transform(pd.Series(X_over.ravel()))
                X_val = vec.transform(X_val)
                nb = MultinomialNB()
                mod = nb.fit(X_over, y_over)
                y_pred_train = nb.predict(X_over)
                y_pred_val = nb.predict(X_val)
                f1_temp = f1_score(y_val, y_pred_val)
                if f1_temp>best_f1_val:
                    best_perc = perc
                    best_f1_val = f1_temp

                    y_pred_val_best = y_pred_val
                    y_over_best = y_over
                    y_pred_train_best = y_pred_train
                    best_model = mod
            except:
                pass
        f1_val = f1_score(y_val, y_pred_val_best)
        precision_val = precision_score(y_val, y_pred_val_best)
        recall_val = recall_score(y_val, y_pred_val_best)
        f1_train = f1_score(y_over_best, y_pred_train_best)
        precision_train = precision_score(y_over_best, y_pred_train_best)
        recall_train = recall_score(y_over_best, y_pred_train_best)
        log_metrics(f1_val, recall_val, precision_val, f1_train, recall_train, precision_train)
        scoring.log_cm(y_over_best, y_pred_train_best, y_val, y_pred_val_best)
        params['sampling_strategy']=round(best_perc/2, 2)
        sv_model(params, best_model)
        endrun()

    for label in y_cols:
        endrun()
        start_mlflow()
        params = params = {
        "model": "NB_oversampling_vs_augmenting",
        "label": label,
        "vectorizer": "tfidf",
        "normalization": "lower",
        "aug_or_os": "aug"}
        best_f1_val=0
        best_perc=0


        for perc in np.arange(0.2, 0.51, 0.1):
            try:
                df_temp = df_train.dropna(subset=[label]).copy()
                df_aug = augmenting.get_augmented_df(df_temp, label, perc)
                X_train= df_aug['text']
                y_train = df_aug[label]
                X_val = df_val['text']
                y_val=  df_val[label]
                vec = TfidfVectorizer(stop_words=stopwords)
                X_train = vec.fit_transform(X_train)
                X_val = vec.transform(X_val)
                nb = MultinomialNB()
                mod = nb.fit(X_train, y_train)
                y_pred_train = nb.predict(X_train)
                y_pred_val = nb.predict(X_val)
                f1_temp = f1_score(y_val, y_pred_val)
                if f1_temp>best_f1_val:
                    best_perc = perc
                    best_f1_val = f1_temp

                    y_pred_val_best = y_pred_val
                    y_train_best = y_train
                    y_pred_train_best = y_pred_train
                    best_model = mod
            except:
                pass
        f1_val = f1_score(y_val, y_pred_val_best)
        precision_val = precision_score(y_val, y_pred_val_best)
        recall_val = recall_score(y_val, y_pred_val_best)
        f1_train = f1_score(y_train_best, y_pred_train_best)
        precision_train = precision_score(y_train_best, y_pred_train_best)
        recall_train = recall_score(y_train_best, y_pred_train_best)
        log_metrics(f1_val, recall_val, precision_val, f1_train, recall_train, precision_train)
        scoring.log_cm(y_train_best, y_pred_train_best, y_val, y_pred_val_best)
        params['sampling_strategy']=round(best_perc, 2)
        sv_model(params, best_model)
        endrun()
