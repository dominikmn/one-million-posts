import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# modeling imports
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from modeling import modeling

if __name__ == "__main__":
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("clf", MultinomialNB()),
    ])
    TARGET_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
        'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
        'label_sentimentnegative', 'label_sentimentpositive',]
    
    data = modeling.Posts()
    training = modeling.Training(data, pipeline)
    for label in TARGET_LABELS:
        data.set_label(label)
        training.train()
        training.evaluate(["train", "val"])
    