# Modeling with Support Vector Machines #32
# Issue link: https://github.com/dominikmn/one-million-posts/issues/32

#import sys
#sys.path.append("..") # add project directory to system path

# general imports
from os import X_OK
import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
import string
import ast
from tqdm import tqdm
import pickle

# NLP imports
#from nltk.tokenize import word_tokenize
#from nltk.stem.cistem import Cistem
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import GermanWortschatzLemmatizer as gwl  #https://docs.google.com/document/d/1rdn0hOnJNcOBWEZgipdDfSyjJdnv_sinuAUSDSpiQns/edit?hl=en#heading=h.oosx9e35prgf
#from nltk.corpus import stopwords
#stopwords=stopwords.words('german')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# modeling imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score

# project utils
from utils import loading, feature_engineering, scoring

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)

# Optional preprocess fuction
def preprocess_snowball(text):
    # normalize (remove capitalization and punctuation)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    words = text.split()

    # stemming
    stemmer = SnowballStemmer(language='german')
    stemmed = [stemmer.stem(w) for w in words]
    
    return stemmed

def train_model(train_val_test, estimator, params, label,):
    logger.info(f"Model-training for label: {label}.")
    X_train = train_val_test['X_train']
    y_train = train_val_test['y_train']

    #### Modeling
    logger.info("Start fitting...")
    estimator.fit(X_train, y_train)

def evaluate_model(train_val_test, estimator, params_base, params_best, label,):
    logger.info(f"Model-evaluation for label: {label}.")

    X_train = train_val_test['X_train']
    X_test = train_val_test['X_test']
    X_val = train_val_test['X_val']
    y_train = train_val_test['y_train']
    y_test = train_val_test['y_test']
    y_val = train_val_test['y_val']

    # setting the MLFlow connection and experiment
    #mlflow.set_tracking_uri(TRACKING_URI)
    #mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        run = mlflow.active_run()
        logger.info("Active run_id: {}".format(run.info.run_id))

        # Predict
        #**TODO** #logger.info(f"Best score: {grid_search.best_score_}")
        y_train_pred = estimator.predict(X_train)
        y_val_pred = estimator.predict(X_val)
        y_test_pred = estimator.predict(X_test)

        #### MlFlow logging
        # Logging params, metrics, and model with mlflow:
        #mlflow.log_metric("train - " + "F1", f1_score(y_train, y_train_pred))
        logger.info(f"train - F1 {f1_score(y_train, y_train_pred)}")
        #mlflow.log_metric("train - " + "recall", recall_score(y_train, y_train_pred))
        #mlflow.log_metric("train - " + "precision", precision_score(y_train, y_train_pred))

        #mlflow.log_metric("val - " + "F1", f1_score(y_val, y_val_pred))
        logger.info(f"val - F1 {f1_score(y_val, y_val_pred)}")
        #mlflow.log_metric("val - " + "recall", recall_score(y_val, y_val_pred))
        #mlflow.log_metric("val - " + "precision", precision_score(y_val, y_val_pred))
        #scoring.log_cm(y_train, y_train_pred, y_val, y_val_pred)

        #mlflow.log_metric("test - " + "F1", f1_score(y_test, y_test_pred))
        #mlflow.log_metric("test - " + "recall", recall_score(y_test, y_test_pred))
        #mlflow.log_metric("test - " + "precision", precision_score(y_test, y_test_pred))
        mlflow_params = dict()
        mlflow_params["normalization"] = 'lower'
        mlflow_params["vectorizer"] = ''.join(e for e in str(vectorizer) if e.isalnum())
        mlflow_params['model'] = ''.join(e for e in str(model) if e.isalnum())
        mlflow_params['grid_search_params'] = params_base
        mlflow_params['label'] = label
        mlflow_params["best_params"] = params_best
        #mlflow.log_params(mlflow_params)
        #mlflow.set_tag("running_from_jupyter", "False")
        #mlflow.log_artifact("./models")
        #mlflow.sklearn.log_model(model, "model")
        t = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = f"models/{mlflow_params['model']}_{label}_{t}"
        #save_model(sk_model=model_best, path=path)

def get_train_val_test(label):
    logger.info("Loading data...")
    try:
        df_train = pd.read_csv(f'./cache/{label}_train.csv', sep='\t')
        df_test = pd.read_csv(f'./cache/{label}_test.csv', sep='\t')
        df_val = pd.read_csv(f'./cache/{label}_val.csv', sep='\t')
    except FileNotFoundError:
        df_train = loading.load_extended_posts(split='train', label=label)
        df_test = loading.load_extended_posts(split='test', label=label)
        df_val = loading.load_extended_posts(split='val', label=label)
        df_train = feature_engineering.add_column_text(df_train)
        df_test = feature_engineering.add_column_text(df_test)
        df_val = feature_engineering.add_column_text(df_val)
        df_train.to_csv(f'./cache/{label}_train.csv', sep='\t')
        df_test.to_csv(f'./cache/{label}_test.csv', sep='\t')
        df_val.to_csv(f'./cache/{label}_val.csv', sep='\t')
    data_dict={
        'X_train': df_train.text,
        'X_val': df_val.text,
        'X_test': df_test.text,

        'y_train': df_train[label],
        'y_val': df_val[label],
        'y_test': df_test[label],
    }
    logger.info("Data loaded.")

    return data_dict

TARGET_LABELS = [
        'label_sentimentnegative', 
        'label_sentimentpositive',
        'label_offtopic', 
        'label_inappropriate', 
        'label_discriminating', 
        'label_possiblyfeedback', 
        'label_personalstories', 
        'label_argumentsused', 
        ]

def load_embedding_vectors(file, embedding_style):
    """
    Args: 
      file: strPath of the embeddingFile.
      embedding_style: {'word2vec', 'glove'}, default=None
    Returns: 
      embedding_dict: dict whose keys are words and the values are the related vectors.
    """
    if embedding_style not in ('word2vec', 'glove'):
       raise ValueError("embedding_style must be any of {'word2vec', 'glove'}") 
    path = f'./cache/embedding_{embedding_style}.pickle'
    try:
        with open(path, 'rb') as f_pickle:
            embedding_dict = pickle.load(f_pickle) 
        logger.info(f"Loaded existing embedding pickle from: {path}.")
    except FileNotFoundError:
        embedding_dict = dict()
        with open(file, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                split_line = line.split()
                if embedding_style == 'word2vec':
                    word = ast.literal_eval(split_line[0]).decode('utf-8')
                elif embedding_style == 'glove':
                    word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                embedding_dict[word] = embedding
            if embedding_style == 'glove':
                embedding_dict['UNK'] = embedding_dict['<unk>']
                del embedding_dict['<unk>']
        with open(path, 'wb') as f_pickle:
            pickle.dump(embedding_dict, file=f_pickle)
        logger.info(f"Computed embedding dictionary and dumped it as pickle: {path}.")
    return embedding_dict
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def fit(self, X, y):
        return self
    
    def _preprocess(self, sentence):
        #**TODO**: Filter-out complete internet-URLs before removing punctuation. 
        # Otherwise the characters and numbers of the URL remain. 
        # However they would all be assigned the "unknown" keyword in transform()
        s = sentence.lower().translate ({ord(c): " " for c in string.punctuation}).split()
        if s==[]: 
            s=['UNK']
        return s
        
    def transform(self, X):
        """
        Args: 
          X: iterable 
            1-dim object containing sentences where the sentence is represented as one string.
        Returns: 
          transformed: np.matrix
            A numpy matrix of the dimension (number of samples, common size of the word vectors)
        """
        transformed = np.matrix([
            np.mean([self.word2vec[w] if w in self.word2vec else self.word2vec['UNK'] for w in self._preprocess(sentence)], axis=0)
            for sentence in X
        ])
        return transformed

if __name__ == '__main__':
    #start=time.time()
    #w2v_dict = load_embedding_vectors('./embeddings/w2v_vectors_orig.txt', embedding_style='word2vec')
    w2v_dict = load_embedding_vectors('./embeddings/glove_vectors_orig.txt', embedding_style='glove')
    #end=time.time()flavoremb_style
    #print(end-start)
    vectorizer = CountVectorizer()
    model = SVC()
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', model),
        ])
    pipeline = Pipeline([
        ('NBCV',MeanEmbeddingVectorizer(w2v_dict)),
        #('nb_norm', MinMaxScaler()),
        ('logreg',LogisticRegression())
    ])

    grid_search_params = {
        #'vectorizer__stop_words': [None], 
        #'vectorizer__preprocessor': [None, preprocess_snowball],
        #'vectorizer__tokenizer': [None],
        'model__C':[.5**0, .5**1, .5**2, .5**3, .5**4, .5**5,],
        'model__kernel':['linear', 'rbf',],
        }
    #grid_search = GridSearchCV(pipeline, param_grid=grid_search_params, cv=5, scoring='f1',
    #                    verbose=3, n_jobs=-1)
    for l in ['label_argumentsused']:#TARGET_LABELS:
        print('-'*50)
        train_val_test = get_train_val_test(l)
        #train_pseudo('BOW', l,)
        train_model(train_val_test, pipeline, grid_search_params, l) 
        evaluate_model(train_val_test, pipeline, grid_search_params, None, l)