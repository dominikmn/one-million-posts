import pandas as pd
import numpy as np
import logging
from typing import Tuple

# evaluation imports
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

# one-million-post utils
from utils import loading, feature_engineering

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME

# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)



def compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, split: str="train"
) -> Tuple[float, float, float]:
    """Computes and logs metrics to mlflow and logger

    Args:
        y_true: The true target classification
        y_pred: The predicted target classification
        split: The split of the dataset ["test", "val", "train"]

    Returns:
        f1: The f1_score
        precision: The precision_score
        recall: The recall_score
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logger.info(
        f"Performance on "
        + str(split)
        + f" set: F1 = {f1:.1f}, precision = {precision:.1%}, recall = {recall:.1%}"
    )
    return f1, precision, recall
            

def predict_with_threshold(y_pred_proba: np.array, threshold: float) -> np.array:
    return (y_pred_proba >= threshold).astype(int)


class Posts:
    AVAILABLE_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
                'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
                'label_sentimentnegative', 'label_sentimentpositive',]
    
    def __init__(self):
        df = loading.load_extended_posts()
        self.df = feature_engineering.add_column_text(df)
        self.current_label = None
    
    def get_X_y(self, split:str=None, label:str=None):
        """Get the features and target variable.
        
        Args:
            label: The label to use as target. Default is all available labels
            split: The data-split to use. One of ['train', 'val', 'test']
        
        Returns:
            X: The feature (text column) of the posts
            y: The target annotations
        """
        if split:
            filter_frame = pd.read_csv(f'./data/ann2_{split}.csv', header=None, index_col=0, names=['id_post'])
            df = self.df.merge(filter_frame, how='inner', on='id_post')
        if label:
            self.current_label = label
            df = df.dropna(subset=[self.current_label])
        elif self.current_label:
            df = df.dropna(subset=[self.current_label])
        else:
            message = f"Please set a label (Post.set_label). Available labels: {self.AVAILABLE_LABELS}"
            logger.error(message)
            raise ValueError(message)
        return df.text, df[self.current_label]
    
    def set_label(self, label:str):
        if label in self.AVAILABLE_LABELS:
            self.current_label = label
        else:
            message = f"Please choose one of the available labels {self.AVAILABLE_LABELS}"
            logger.error(message)
            raise ValueError(message)


class Training:
    """
    Args:
        data:
        estimator:
    """
    
    def __init__(self, data:Posts, estimator):
        self.data = data
        self.estimator = estimator
        self.model = None
        self.threshold = None
        
    
    def calculate_best_threshold(self, y_true: pd.Series, y_pred_proba: np.array) -> float:
        """Calculate the best threshold value for classification.

        Args:
            y_pred_proba: Predicted probabilities.
            y_true: Series with true target labels.

        Returns:
            best_th: The threshold value that maximizes the f1-score.
        """
        best_th = 0.0
        best_f1 = 0.0
        for th in np.arange(0.05, 0.96, 0.05):
            y_pred_temp = predict_with_threshold(y_pred_proba, th)
            f1_temp = f1_score(y_true, y_pred_temp)
            if f1_temp > best_f1:
                best_th = th
                best_f1 = f1_temp
        return best_th


    def save_model_to_drive(self, model):
        """
        """
        # ToDo get model info
        logger.info("Saving model in the models folder")
        t = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = f"models/{model_details['name']}_{mlflow_params['label']}_{t}"
        save_model(sk_model=model, path=path)


    def train(self): #, mlflow_logger):
        """Trains the estimator.

        If model implements predict_proba, calculate the best cut-off threshold.
        If the estimator is a GridSearch, it stores the best_params_ in mlflow_params and returns the best_model_ as model. Stopwords must be set via `vectorizer__stop_words`.

        Args:
            estimator:
            mlflow_logger:

        Returns:
            model:
            mlflow_params:
            threshold:
        """
        mlflow_params = {} #ToDo fix mlflow

        X_train, y_train = self.data.get_X_y(split="train")
        
        logger.info(f"Fit model")
        model = self.estimator.fit(X_train, y_train) #ToDo save to mlflow logger

        # select best threshold if model implements predict_proba
        if callable(getattr(model, "predict_proba", None)):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            self.threshold = self.calculate_best_threshold(y_train, y_train_proba)
            mlflow_params["threshold"] = self.threshold #ToDo save to mlflow logger

        # store best parameters if model is GridSearch
        if isinstance(model, GridSearchCV):
            best_params = model.best_params_
            if 'vectorizer__stop_words' in best_params.keys() and best_params['vectorizer__stop_words']!=None:
                best_params['vectorizer__stop_words'] = "NLTK-German"
            mlflow_params["best_params"] = best_params #ToDo save to mlflow logger

        self.model = model
        #self.save_model_to_drive(model)
        return model


    def evaluate(self, splits=["train", "val"]):
        """Calculate predictions and metrics.

        """
        if self.model:
            for split in splits:
                X, y = self.data.get_X_y(split=split)

                model = self.model

                if self.threshold:
                    y_pred = predict_with_threshold(model.predict_proba(X)[:, 1], self.threshold)
                else:
                    y_pred = model.predict(X)

                compute_and_log_metrics(y, y_pred, split)
        else:
            logger.info(f"No model available! Please run `train()` first.")

