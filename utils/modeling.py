import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict

# evaluation imports
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# one-million-post utils
from utils import loading, feature_engineering, augmenting

# mlflow
import mlflow
from mlflow.sklearn import save_model


# set logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)



def compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, split: str="train"
) -> Tuple[float, float, float]:
    """Computes and logs metrics logger

    Args:
        y_true: The true target classification
        y_pred: The predicted target classification
        split: The split of the dataset ["test", "val", "train"]

    Returns:
        f1: The f1_score
        precision: The precision_score
        recall: The recall_score
        cm: Dictionary with the confusion matrix
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
    """Handles the data.

    Attributes:
        split: The data-split to use. One of [None, 'train', 'val', 'test']. If None, the whole
            dataset is returned
        current_label: The label to use as target.
        balance_method: The balancing method. One of [None, 'translate', 'oversample']
        sampling_strategy: The desired ratio of the number of samples in the minority class
            over the number of samples in the majority class after resampling/augmenting.
    """
    AVAILABLE_LABELS = ['label_argumentsused', 'label_discriminating', 'label_inappropriate',
                'label_offtopic', 'label_personalstories', 'label_possiblyfeedback',
                'label_sentimentnegative', 'label_sentimentpositive',]
    
    def __init__(self):
        df = loading.load_extended_posts()
        self.df = feature_engineering.add_column_text(df)
        self.current_label = None
        self.balance_method = None
        self.sampling_strategy = 1

    def get_X_y(self, split:str=None, label:str=None, balance_method:str=None, sampling_strategy:float=None) -> Tuple[pd.Series, pd.Series]:
        """Get the features and target variable.

        label, balance_method, and sampling_strategy overwrite the attributes of Post if given.

        Args:
            split: The data-split to use. One of [None, 'train', 'val', 'test']. If None, the whole
                dataset is returned
            label: The label to use as target.
            balance_method: The balancing method. One of ['translate', 'oversample']
            sampling_strategy: The desired ratio of the number of samples in the minority class
                over the number of samples in the majority class after resampling/augmenting.

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
        try:
            df = df.dropna(subset=[self.current_label])
        except KeyError as e:
            message = f"Please set a label (Post.set_label). Available labels: {self.AVAILABLE_LABELS}"
            logger.error(message)
            raise KeyError(e)

        X = df.text
        y = df[self.current_label]

        if balance_method:
            self.balance_method = balance_method
        if sampling_strategy:
            self.sampling_strategy = sampling_strategy
        if self.balance_method is "translate":
            X, y = augmenting.get_augmented_X_y(X, y, label=self.current_label, sampling_strategy=self.sampling_strategy)
        elif self.balance_method is "oversample":
            X, y = augmenting.get_oversampled_X_y(X, y, sampling_strategy=self.sampling_strategy)
        return X, y

    def set_label(self, label:str):
        if label in self.AVAILABLE_LABELS:
            self.current_label = label
        else:
            message = f"Please choose one of the available labels {self.AVAILABLE_LABELS}"
            logger.error(message)
            raise ValueError(message)

    def set_balance_method(self, balance_method, sampling_strategy):
        self.balance_method = balance_method
        self.sampling_strategy = sampling_strategy


class MLFlowLogger:
    """MLFlowLogger will collect all params, tags, and metrics to log.
    """
    def __init__(self, uri:str, experiment:str, is_dev:bool=True,
        params: Dict=dict(), tags: Dict=dict(), metrics: Dict=dict()):
        self.is_dev = is_dev
        self.model_path = Path("./models")
        self.params = params if params else {}
        self.tags = tags if tags else {}
        self.metrics = metrics if metrics else {}
        self.model = None
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    def add_param(self, key:str, value:str):
        self.params[key] = value

    def add_tag(self, name:str, state:bool):
        self.tags[name] = state

    def add_metric(self, name:str, value):
        self.metrics[name] = value

    def log(self):
        """Log params, metrics, and tags to MLFlow if is_def is False"""
        if not self.is_dev:
            mlflow.log_params(self.params)
            mlflow.log_metrics(self.metrics)
            mlflow.set_tags(self.tags)
            self._save_model()

    def _save_model(self):
        """Save model in `./models` as `<model_type>_<label>_<time>`
        """
        if self.model:
            logger.info(f"Saving model in {self.model_path}.")
            time_now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            model_type = self.params["model"]
            label = self.params["label"]
            path = self.model_path / f"{model_type}_{label}_{time_now}"
            save_model(sk_model=self.model, path=path)


class Modeling:
    """Class to handle the modeling with training and evaluation.

    Args:
        data: A Post object handling the data
        estimator: An sklearn.estimator
        mlflow_logger: An MLFlowLogger object collecting all params, metrics, and tags.
        fit_threshold: Fit a best decision threshold if True. Optional, default: True
    """
    def __init__(self, data:Posts, estimator, mlflow_logger, fit_threshold:bool=True):
        self.data = data
        self.estimator = estimator
        self.mlflow_logger = mlflow_logger
        self.fit_threshold = fit_threshold
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
        If the estimator is a GridSearch, it stores the best_params_ in mlflow_params
        and returns the best_model_ as model. **Stopwords must be set via `vectorizer__stop_words`**.

        Returns:
            model: A trained estimator
        """
        logger.info(f"Get X, y")
        X_train, y_train = self.data.get_X_y(split="train")
        self.mlflow_logger.add_param("label", self.data.current_label)
        self.mlflow_logger.add_param("balance_method", self.data.balance_method)
        if self.data.balance_method:
            self.mlflow_logger.add_param("sampling_strategy", self.data.sampling_strategy)
        
        logger.info(f"Fit model")
        self.estimator.fit(X_train, y_train) #ToDo save to mlflow logger

        # select best threshold if model implements predict_proba
        if self.fit_threshold and callable(getattr(self.estimator, "predict_proba", None)):
            y_train_proba = self.estimator.predict_proba(X_train)[:, 1]
            self.threshold = self.calculate_best_threshold(y_train, y_train_proba)
            self.mlflow_logger.add_param("threshold", self.threshold)

        # store best parameters if model is GridSearch
        if isinstance(self.estimator, GridSearchCV):
            best_params = self.estimator.best_params_
            if 'vectorizer__stop_words' in best_params.keys() and best_params['vectorizer__stop_words']!=None:
                best_params['vectorizer__stop_words'] = "NLTK-German"
            for k, v in best_params.items():
                self.mlflow_logger.add_param(f"best_{k}", v)

        self.model = self.estimator
        return self.model


    def evaluate(self, splits=["train", "val"]):
        """Calculate predictions and metrics.

        Args:
            splits: List of splits to evaluate. Default: ["train", "val"]
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

