# +
import pandas as pd

def create_label_csv(label_list):
    df = pd.DataFrame()
    for label in label_list:
        path = f"./output/trans_{label}.csv"
        temp = pd.read_csv(path)
        df = pd.concat((df, temp))   

    return df

def add_column(df, column_name):
    df[column_name]=1
    return df


# -

if __name__=='__main__':
    
    label_list =['label_discriminating', 'label_inappropriate', 'label_sentimentnegative', 'label_offtopic']
    df = create_label_csv(label_list, 'train')
    df = add_column(df, 'label_needsmoderation')
    df.to_csv('./output/trans_label_needsmoderation.csv')
        
        
    label_list =['label_discriminating', 'label_inappropriate', 'label_sentimentnegative']
    df = create_label_csv(label_list, 'train')
    df = add_column(df, 'label_negative')
    df.to_csv('./output/trans_label_negative.csv')