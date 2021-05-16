import pandas as pd
import numpy as np
from utils import loading, feature_engineering
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
        
def get_augmented_val(df, label):
    '''get a dataset with augmented texts for the minority positive label
    Arguments: X, y - pandas series containing the validation data that needs to be augmented
               label - label that needs to be augmented
               sampling_strategy - float representing the proportion of positive vs negative labels in the augmented dataframe (range [>0.0; <=1.ß])
    Return: augmented X, y'''
    
    try:
        df_aug = pd.read_csv(f'./output/trans_val_{label}.csv')
        return pd.concat((df, df_aug))
    except FileNotFoundError as e:
        print(f'Requested augmentation data for label {label} not available. Returned original df')
        return df


def get_augmented_val_X_y(X, y, label):
    '''get a dataset with augmented texts for the minority positive label
    Arguments: X, y - pandas series containing the validation data that needs to be augmented
               label - label that needs to be augmented
               sampling_strategy - float representing the proportion of positive vs negative labels in the augmented dataframe (range [>0.0; <=1.ß])
    Return: augmented X, y'''
    
    label_range = ['label_sentimentnegative', 'label_inappropriate', 'label_discriminating', 'label_offtopic','label_needsmoderation', 'label_negative']
    
    if label in label_range:
        file_cached = "./cache/df_r3.csv"
        
        try:
            df_aug = pd.read_csv(f'./output/trans_val_{label}.csv')
            X_aug = df_aug['text']
            y_aug = df_aug[label]
            X, y = pd.concat((X, X_aug)), pd.concat((y, y_aug, ))
        
        except FileNotFoundError as e:
            pass
        try:
             df_r3 = pd.read_csv(file_cached)

        except:
            df_r3 = loading.load_extended_posts(label=label)
            df_r3 = feature_engineering.add_column_ann_round(df_r3)
            df_r3 = feature_engineering.add_column_text(df_r3)
            df_r3 = df_r3.query('ann_round==3').copy()
            df_r3.to_csv(file_cached)

        df_r3 = feature_engineering.add_column_label_needsmoderation(df_r3)
        art_list = list(df_r3.id_article.unique())
        df_ann = pd.DataFrame(columns=df_r3.columns)

        for i in art_list:
            df_ann = pd.concat((df_ann,
                                df_r3.query(f'id_article=={i} and {label}==1').sample(1, 
                                                                                      random_state=42)))

        return  pd.concat((X,  df_ann['text'])), pd.concat((y, df_ann[label] ))
   
    else:
        print(f'Requested augmentation data for label {label} not available. Returned original X,y')
        return X, y



def get_augmented_val_id():
    '''get a dataset with augmented texts for the minority positive label
    Arguments: X, y - pandas series containing the validation data that needs to be augmented
               label - label that needs to be augmented
               sampling_strategy - float representing the proportion of positive vs negative labels in the augmented dataframe (range [>0.0; <=1.ß])
    Return: augmented X, y'''
    
    label_range = ['label_sentimentnegative', 'label_inappropriate', 'label_discriminating', 'label_needsmoderation']
    file_cached = "./cache/df_r3.csv"
    try:
         df_r3 = pd.read_csv(file_cached)

    except:
        df_r3 = loading.load_extended_posts(label=label)
        df_r3 = feature_engineering.add_column_ann_round(df_r3)
        df_r3 = feature_engineering.add_column_text(df_r3)
        df_r3 = df_r3.query('ann_round==3').copy()
        df_r3.to_csv(file_cached)

    df_r3 = feature_engineering.add_column_label_needsmoderation(df_r3)
    art_list = list(df_r3.id_article.unique())
    
    label_range = ['label_sentimentnegative', 'label_inappropriate', 'label_discriminating', 'label_needsmoderation']
    df_ann = pd.DataFrame(columns=df_r3.columns)
    
    id_list = []
    for label in label_range:
        for i in art_list:
            df_ann = pd.concat((df_ann,
                                df_r3.query(f'id_article=={i} and {label}==1').sample(1, 
                                                                                      random_state=42)))
            id_list.extend(list(df_ann.id_post))

    return  list(set(id_list))