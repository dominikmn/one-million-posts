import pandas as pd
from sklearn.model_selection import train_test_split

from utils import loading, feature_engineering
 

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
    
    # The first round of EDA showed, that only posts annotated in round 2 represent the population. 
    # Posts annotated in round 3 will only be used for the labels "possiblyFeedback" and 
    # "personalStories" (i.e. NaN in any of the other lables) and only in the training set.
    df_ann2 = df_posts.query("ann_round == 2")
    df_ann3_feedback_stories = df_posts.query("ann_round == 3 and label_offtopic != label_offtopic")
    
    # Due to a small dataset (1,000 posts) we want to keep 100 observations for test and validation split each
    # We stratify by labels, that are least frequent in our 1,000 observations
    features_strat = ['label_discriminating', 'label_possiblyfeedback', 'label_personalstories', 'label_sentimentpositive']
    ann2_train, ann2_test = train_test_split(df_ann2, stratify=df_ann2[features_strat],random_state=RSEED, test_size=100)
    ann2_train, ann2_val = train_test_split(ann2_train, stratify=ann2_train[features_strat],random_state=RSEED, test_size=100)

    df_train = pd.concat([ann2_train, df_ann3_feedback_stories], axis=0)

    print(f"Number of posts in train-set: {df_train.shape[0]}")
    print(f"Number of posts in val-set: {ann2_val.shape[0]}")
    print(f"Number of posts in test-set: {ann2_test.shape[0]}")
    df_train.id_post.to_csv('./data/ann2_train.csv', header=False)
    ann2_test.id_post.to_csv('./data/ann2_test.csv', header=False)
    ann2_val.id_post.to_csv('./data/ann2_val.csv', header=False)
    print('Splits created.')


if __name__ == '__main__':
    create_splits()
