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
    param_grid = {
        "vectorizer__ngram_range" : [(1,1), (1,2), (1,3)],
        "vectorizer__stop_words" : [stopwords, None],
        "vectorizer__min_df": np.linspace(0, 0.1, 3),
        "vectorizer__max_df": np.linspace(0.9, 1.0, 3),
    }
    gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=3, verbose=1)

    TARGET_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
        'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
        'label_sentimentnegative', 'label_sentimentpositive',]
    
    data = modeling.Posts()
    training = modeling.Training(data, gs)
    for label in TARGET_LABELS[:1]:
        data.set_label(label)
        training.train()
        training.evaluate(["train", "val"])
    