import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
from typing import Tuple, Dict
import logging


# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)


category_map = {
                 'argumentation': 'label_argumentsused',
                 'discriminating': 'label_discriminating',
                 'inappropriate': 'label_inappropriate',
                 'a personal story': 'label_personalstories',
                 'off-topic': 'label_offtopic',
                 'requiring feedback': 'label_possiblyfeedback',
                 'positive': 'label_sentimentpositive',
                 'neutral': 'label_sentimentneutral',
                 'negative': 'label_sentimentnegative'
                }

def baseline_df():
    return pd.DataFrame([
             {
             'label':'label_sentimentnegative',
             'precision': 0.6216, 
             'recall': 0.6014, 
             'f1': 0.6063},
                     {
             'label': 'label_sentimentneutral',
             'precision': 1.0, 
             'recall': 1.0, 
             'f1': 1.0},
                     {
             'label':'label_sentimentpositive',
             'precision': 0.1020, 
             'recall': 0.4651, 
             'f1': 0.1579},
                     {
             'label':'label_offtopic',
             'precision': 0.2579, 
             'recall': 0.6241, 
             'f1': 0.3516},
                     {
             'label':'label_inappropriate',
             'precision': 0.1475, 
             'recall': 0.5974, 
             'f1': 0.2175},
                     {
             'label':'label_discriminating',
             'precision': 0.1547, 
             'recall': 0.5922, 
             'f1': 0.2005},
                     {
             'label': 'label_possiblyfeedback',
             'precision': 0.5393, 
             'recall': 0.7679, 
             'f1': 0.6168},
                     {
             'label':'label_personalstories',
             'precision': 0.6247, 
             'recall': 0.8505, 
             'f1': 0.7063},
                     {
             'label':'label_argumentsused',
             'precision': 0.5657, 
             'recall': 0.7769, 
             'f1': 0.6371}
    ])

def get_score_df(df, best = False, threshold = 0.5):
    score = []  
    for label in category_map.values():
        lab_true = label+'_true'
        lab_pred = label+'_pred'
        y_true = df[lab_true]
        if best:
            best_th = 0.0
            best_f1 = 0.0
            for th in np.arange(0.05, 0.96, 0.05):
                y_pred_temp = df[lab_pred].apply(lambda x: 1 if x>=th else 0)
                if f1_score(y_true, y_pred_temp)>best_f1:
                    best_th = th
                    best_f1 = f1_score(y_true, y_pred_temp)
        else:
            best_th = threshold
        y_pred = df[lab_pred].apply(lambda x: 1 if x>=best_th else 0)
        score.append({
                            'label':label,
                            'threshold': best_th,
                            'precision': precision_score(y_true, y_pred), 
                            'recall': recall_score(y_true, y_pred), 
                            'f1': f1_score(y_true, y_pred)})
        baseline = baseline_df()
        score_data = pd.merge(pd.DataFrame(score), baseline, how = 'left', on = 'label', suffixes = ('_pred', '_base'))
        score_data['precision_win'] = (score_data['precision_pred']-score_data['precision_base']).apply(lambda x: 1 if x>0 else 0)
        score_data['recall_win'] = (score_data['recall_pred']-score_data['recall_base']).apply(lambda x: 1 if x>0 else 0)
        score_data['f1_win'] = (score_data['f1_pred']-score_data['f1_base']).apply(lambda x: 1 if x>0 else 0)
        score_data['we_have_a_winner'] = (score_data['precision_win'] + score_data['recall_win'] + score_data['f1_win']).apply(lambda x: 1 if x==3 else 0)
    return score_data

def print_winners(df):
    f1_winners = list(df.query('f1_win==1').label)
    all_winners = list(df.query('we_have_a_winner==1').label)
    if len(f1_winners)>=1:
        print('The tested model beats the baseline in f1 score in the following labels:\n')
        for winner in f1_winners:
            print(winner)
            print()
    if len(all_winners)>=1:
        print('The tested model beats the baseline in all metrics in the following labels:\n')
        for winner in all_winners:
            print(winner)
            print()    

            
def log_cm (y_true_train, y_pred_train, y_true_val, y_pred_val):
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_true_train, y_pred_train).ravel()
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_true_val, y_pred_val).ravel()
    cm_tr = {'TN':tn_tr, 'FP':fp_tr, 'FN':fn_tr, 'TP':tp_tr}
    cm_val = {'TN':tn_val, 'FP':fp_val, 'FN':fn_val, 'TP':tp_val}
    mlflow.log_params({"cm-train": cm_tr, "cm-val": cm_val})


def compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, split: str="train"
) -> Tuple[float, float, float, Dict]:
    """Computes and logs metrics logger

    Args:
        y_true: The true target classification
        y_pred: The predicted target classification
        split: The split of the dataset ["test", "val", "train"]

    Returns:
        f2: The f2_score
        f1: The f1_score
        precision: The precision_score
        recall: The recall_score
        cm: Dictionary with the confusion matrix
    """
    f2 = fbeta_score(y_true, y_pred, beta=2)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = {'TN':tn, 'FP':fp, 'FN':fn, 'TP':tp}

    logger.info(f"Performance on {split} set: F2 = {f2:1f} F1 = {f1:.1f}, precision = {precision:.1%}, recall = {recall:.1%}")
    logger.info(f"Confusion matrix: {cm}")
    return f2, f1, precision, recall, cm
