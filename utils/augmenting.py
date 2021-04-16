import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def get_augmented_X_y(X, y, sampling_strategy, label):
    '''get a dataset with augmented texts for the minority positive label
    Arguments: X, y - pandas series containing the training data that needs to be augmented
               label - label that needs to be augmented
               sampling_strategy - float representing the proportion of positive vs negative labels in the augmented dataframe (range [>0.0; <=1.ß])
    Return: augmented X, y'''
    perc = sampling_strategy/2
    pos = y.value_counts()[1]
    neg = y.value_counts()[0]
    tot = pos+neg
    try:
        df_aug = pd.read_csv(f'./output/trans_{label}.csv')
    except FileNotFoundError as e:
        print(f'Requested augmentation data for label {label} not available. Returned original X and y')
        return X, y
    samp_n = int((pos - perc*tot)/(perc-1))
    if samp_n<0:
        print('Requested percentage is lower than original percentage. Returned original X and y')
        return X, y
    else:
        samp = df_aug.sample(n=samp_n, random_state=42)
        X_aug = samp['text']
        y_aug = samp[label]
        return pd.concat((X, X_aug)), pd.concat((y, y_aug))

    
def get_oversampled_X_y(X, y, sampling_strategy):
    '''get a dataset with oversampled texts for the minority positive label
    Arguments: X, y - pandas series containing the training data that needs to be augmented
               label - label that needs to be augmented
               sampling_strategy - float representing the proportion of positive vs negative labels in the oversampled dataframe (range [>0.0; <=1.ß])
    Return: oversampled X, y'''
    perc = sampling_strategy/2
    pos = y.value_counts()[1]
    neg = y.value_counts()[0]
    tot = pos+neg
    samp_n = int((pos - perc*tot)/(perc-1))
    if samp_n<0:
        print('Requested percentage is lower than original percentage. Returned original X and y')
        return X, y
    else:
        X_temp= np.array(X).reshape(-1, 1)
        y_temp = np.array(y).reshape(-1, 1)
        os = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_over, y_over = os.fit_resample(X_temp, y_temp)
        return pd.Series(X_over.ravel()), pd.Series(y_over.ravel())
        