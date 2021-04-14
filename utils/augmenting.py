import pandas as pd

def num_samp(pos, ges, perc):
    return int((pos - perc*ges)/(perc-1))

def get_aug_os(df, label, perc):
    
    pos = df[label].value_counts()[1]
    neg = df[label].value_counts()[0]
    ges = pos+neg
    df_aug = pd.read_csv(f'./output/trans_{label}.csv')
    samp_n = num_samp(pos, ges, perc)
    if samp_n<0:
        print('Requested percentage is lower than original percentage. Returned original df')
        return df
    else:
        samp = df_aug.sample(n=samp_n, random_state=42)
        return pd.concat((df, samp))
