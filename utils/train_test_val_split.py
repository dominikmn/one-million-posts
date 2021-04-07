from utils import loading, feature_engineering
from sklearn.model_selection import train_test_split 

def create_splits():
    """
    Create test-train-val split of labeled data and save it to csv.
    Annotation round 2 and round 3 are split seperately.
    The csv files are written to ./data and contain the id_post.
    Args: None
    Returns: None
    """
    RSEED=42
    df_posts = loading.load_extended_posts() 
    df_posts = feature_engineering.add_column_ann_round(df_posts)
    df_ann2 = df_posts.query("ann_round == 2")
    df_ann3 = df_posts.query("ann_round == 3")
    features_strat = ['label_discriminating', 'label_inappropriate','label_sentimentpositive']
    
    ann2_train, ann2_test = train_test_split(df_ann2, stratify=df_ann2[features_strat],random_state=RSEED, test_size=250)
    ann2_train, ann2_val = train_test_split(ann2_train, stratify=ann2_train[features_strat],random_state=RSEED, test_size=100)

    #ann3_train, ann3_test = train_test_split(df_ann3, stratify=df_ann3[features_strat],random_state=RSEED, test_size=0.25)
    #ann3_train, ann3_val = train_test_split(ann3_train, stratify=ann3_train[features_strat],random_state=RSEED, test_size=0.13)

    ann2_train.id_post.to_csv('./data/ann2_train.csv', header=False)
    ann2_test.id_post.to_csv('./data/ann2_test.csv', header=False)
    ann2_val.id_post.to_csv('./data/ann2_val.csv', header=False)

    #ann3_train.id_post.to_csv('./data/ann3_train.csv', header=False)
    #ann3_test.id_post.to_csv('./data/ann3_test.csv', header=False)
    #ann3_val.id_post.to_csv('./data/ann3_val.csv', header=False)

    print('Splits created.')

if __name__ == '__main__':
    create_splits()
