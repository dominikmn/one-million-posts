import pandas as pd


def get_augmented_df(df, label, perc):
    '''get a dataset with augmented texts for the minority positive label
    Arguments: df - pandas dataframe containing the training data that needs to be augmented
               label - label that needs to be augmented
               perc - float representing the proportion of positive labels in the augmented dataframe (range [>0.0; <=0.5])
    Return: an augmented dataframe'''
    
    pos = df[label].value_counts()[1]
    neg = df[label].value_counts()[0]
    tot = pos+neg
    try:
        df_aug = pd.read_csv(f'./output/trans_{label}.csv')
    except FileNotFoundError as e:
        print(f'Requested augmentation data for label {label} not available. Returned original df')
        return df
    samp_n = int((pos - perc*tot)/(perc-1))
    if samp_n<0:
        print('Requested percentage is lower than original percentage. Returned original df')
        return df
    else:
        samp = df_aug.sample(n=samp_n, random_state=42)
        return pd.concat((df, samp))
