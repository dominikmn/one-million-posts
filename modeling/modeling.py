import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict

# evaluation imports
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# one-million-post utils
from utils import loading, feature_engineering

# mlflow
import mlflow
from mlflow.sklearn import save_model
from modeling.config import TRACKING_URI, EXPERIMENT_NAME#, TRACKING_URI_DEV

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
        cm:
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = {'TN':tn, 'FP':fp, 'FN':fn, 'TP':tp}

    logger.info(f"Performance on {split} set: F1 = {f1:.1f}, precision = {precision:.1%}, recall = {recall:.1%}")
    logger.info(f"Confusion matrix: {cm}")
    return f1, precision, recall, cm
            

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
        else:
            df = self.df.copy()
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


class MLFlowLogger:
    def __init__(self, uri:str=None, experiment:str=None, is_dev:bool=True, params: Dict=dict(), tags: Dict=dict(), metrics: Dict=dict()):
        self.is_dev = is_dev
        self.model_path = Path("./models")
        self.params = params if params else {}
        self.tags = tags if tags else {}
        self.metrics = metrics if metrics else {}
        self.model = None
        #if is_dev:
            #uri = TRACKING_URI_DEV
            #self.model_path = self.model_path / "dev"
        if not uri:
            uri = TRACKING_URI
        if not experiment:
            experiment = EXPERIMENT_NAME
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    def add_param(self, key:str, value:str):
        self.params[key] = value

    def add_tag(self, name:str, state:bool):
        self.tags[name] = state

    def add_metric(self, name:str, value):
        self.metrics[name] = value

    def log(self):
        if not self.is_dev:
            mlflow.log_params(self.params)
            mlflow.log_metrics(self.metrics)
            mlflow.set_tags(self.tags)
            self._save_model()

    def _save_model(self):
        """
        """
        if self.model:
            logger.info(f"Saving model in {self.model_path}.")
            time_now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            model_type = self.params["model"]
            label = self.params["label"]
            path = self.model_path / f"{model_type}_{label}_{time_now}"
            save_model(sk_model=self.model, path=path)


class Training:
    """
    Args:
        data:
        estimator:
    """
    def __init__(self, data:Posts, estimator, mlflow_logger):
        self.data = data
        self.estimator = estimator
        self.mlflow_logger = mlflow_logger
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
        logger.info(f"Get X, y")
        X_train, y_train = self.data.get_X_y(split="train")
        self.mlflow_logger.add_param("label", self.data.current_label)
        
        logger.info(f"Fit model")
        self.estimator.fit(X_train, y_train) #ToDo save to mlflow logger

        # select best threshold if model implements predict_proba
        if callable(getattr(self.estimator, "predict_proba", None)):
            y_train_proba = self.estimator.predict_proba(X_train)[:, 1]
            self.threshold = self.calculate_best_threshold(y_train, y_train_proba)
            self.mlflow_logger.add_param("threshold", self.threshold)

        # store best parameters if model is GridSearch
        if isinstance(self.estimator, GridSearchCV):
            best_params = self.estimator.best_params_
            if 'vectorizer__stop_words' in best_params.keys() and best_params['vectorizer__stop_words']!=None:
                best_params['vectorizer__stop_words'] = "NLTK-German"
            self.mlflow_logger.add_param("best_params", best_params)

        self.model = self.estimator
        return self.model


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

                f1, precision, recall, cm = compute_and_log_metrics(y, y_pred, split)
                self.mlflow_logger.add_metric(f"{split} - F1", f1)
                self.mlflow_logger.add_metric(f"{split} - precision", precision)
                self.mlflow_logger.add_metric(f"{split} - recall", recall)
                self.mlflow_logger.add_param(f"cm-{split}", cm)
        else:
            logger.info(f"No model available! Please run `train()` first.")

