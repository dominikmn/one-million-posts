# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # German BERT - minimal setup #22 
# Issue link: https://github.com/dominikmn/one-million-posts/issues/22

# %% [markdown]
# The pre-trained model can be found under:
# https://huggingface.co/deepset/gbert-base
#
# The code below was initially built up with the help of this tutorial: https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

# %%
from os import stat
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from datetime import datetime

from utils import loading, feature_engineering, augmenting, scoring, cleaning
from utils import modeling as m

import mlflow
from modeling.config import TRACKING_URI, EXPERIMENT_NAME#, TRACKING_URI_DEV

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)

# %% [markdown] tags=[]
# ## Global params

# %%
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

# %%
logger.info(f'Computations will take place on: {device}')


# %% [markdown]
# ## Definitions

# %%
class OMPDataset(Dataset):
    def __init__(self, text, targets, tokenizer, max_len):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            return_attention_mask=True,
            return_tensors='pt',
    )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
    }


# %%
def create_data_loader(df:pd.DataFrame, label:str, tokenizer:BertTokenizer, max_len:int, batch_size:int) -> DataLoader:
    ds = OMPDataset(
        text=df.text.to_numpy(),
        targets=df[label].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


# %%
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        ).values()
        output = self.dropout(pooled_output)
        output = self.out(output)
        return output

# %%
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler,):
    model = model.train()
    losses = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for d in data_loader:
        # **TODO** This code block is completely identical between eval_model and train_epoc
        # START of code block
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        targets = targets.unsqueeze(1)
        targets = targets.float()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        preds = torch.sigmoid(outputs)
        preds = torch.round(preds)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        tp += (targets * preds).sum(dim=0).to(torch.float32)
        tn += ((1 - targets) * (1 - preds)).sum(dim=0).to(torch.float32)
        fp += ((1 - targets) * preds).sum(dim=0).to(torch.float32)
        fn += (targets * (1 - preds)).sum(dim=0).to(torch.float32)
        # END of code block

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    split = 'train'
    fbeta, metrics, params = _give_scores(tp, tn, fp, fn, split)

    return fbeta, metrics, params, np.mean(losses)

# %%
def eval_model(model, data_loader, loss_fn, device, split: str):
    model = model.eval()
    losses = []
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for d in data_loader:
            # **TODO** This code block is completely identical between eval_model and train_epoc
            # START of code block
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            targets = targets.unsqueeze(1)
            targets = targets.float()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
            preds = torch.sigmoid(outputs)
            preds = torch.round(preds)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            tp += (targets * preds).sum(dim=0).to(torch.float32)
            tn += ((1 - targets) * (1 - preds)).sum(dim=0).to(torch.float32)
            fp += ((1 - targets) * preds).sum(dim=0).to(torch.float32)
            fn += (targets * (1 - preds)).sum(dim=0).to(torch.float32)
            # END of code block

    split = split
    fbeta, metrics, params = _give_scores(tp, tn, fp, fn, split)

    return fbeta, metrics, params, np.mean(losses)

# %%
def _give_scores(tp, tn, fp, fn, split):
    ''' Helper function to compute recall, precision, f1, fbeta from tensors.

    '''
    # Helper-function to retrieve float from tensor:
    z = lambda x: float(x.data.cpu().numpy()[0])

    eps=1E-5

    f1 = tp / (tp + 0.5 * (fp + fn) + eps)
    beta = 2.
    fbeta = ((1. + beta**2) * tp) / ((1. + beta**2) * tp + (beta**2) * fn + fp + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    cm = {'TN':int(z(tn)), 'FP':int(z(fp)), 'FN':int(z(fn)), 'TP':int(z(tp))}
    
    if split in ['train', 'val']:
        name = f"{split}-bal"
    else:
        name = f"{split}"
    metrics = dict()
    metrics[f"{name} - F1"] = z(f1)
    metrics[f"{name} - F2"] = z(fbeta)
    metrics[f"{name} - precision"] =  z(precision)
    metrics[f"{name} - recall"] = z(recall)
    
    params = dict()
    params[f"cm-{name}"] =  cm
    
    return fbeta,metrics,params

# %% [markdown]
# ## Main method

# %%
def make_model(data:m.Posts, label:str, learning_rate:float, positive_class_weight:float):
    BATCH_SIZE = 8
    MAX_LEN = 264
    EPOCHS = 10

    # ## MLflow instantiation
    IS_DEVELOPMENT = False
    mlflow_logger = m.MLFlowLogger(
        uri=TRACKING_URI,
        experiment=EXPERIMENT_NAME,
        is_dev=IS_DEVELOPMENT,
    )

    # ## Data loading
    normalize = lambda x: cleaning.normalize(x, url_emoji_dummy=False, pure_words=False)

    X_train, y_train = data.get_X_y('train')
    X_train = X_train.apply(normalize)
    df_train = pd.concat([X_train,y_train], axis=1)
    df_train.columns = ['text', label]

    X_val, y_val = data.get_X_y('val', balance_method='translate')
    X_val = X_val.apply(normalize)
    df_val = pd.concat([X_val,y_val], axis=1)
    df_val.columns = ['text', label]

    #X_test, y_test = data.get_X_y('test')
    #X_test = X_test.apply(normalize)
    #df_test = pd.concat([X_test,y_test], axis=1)
    #df_test.columns = ['text', label]

    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    train_data_loader = create_data_loader(df_train, label, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, label, tokenizer, MAX_LEN, BATCH_SIZE)
    #test_data_loader = create_data_loader(df_test, label, tokenizer, MAX_LEN, BATCH_SIZE)

    # ## Instantiation
    model = BinaryClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    mlflow_logger.add_param('optimizer', 'AdamW')
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
        )
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    # From the BCEWithLogitLoss() documentation:
    # > For example, if a dataset contains 100 positive and 300 negative examples of a single class, 
    # > then pos_weight for the class should be equal to 300/100=3 . 
    # > The loss would act as if the dataset contains 3*100=300 positive examples.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([positive_class_weight])).to(device)

    history = defaultdict(list)
    best_fbeta = 0
    t = datetime.now().strftime("%y%m%d_%H%M%S")
    for epoch in range(EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}')
        logger.info('-' * 10)
        train_fbeta, train_metrics, train_params, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            'train',
        )
        logger.info(f'Train loss {train_loss} Fbeta {train_fbeta}')
        val_fbeta, val_metrics, val_params,  val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            'val',
        )
        logger.info(f'Val   loss {val_loss} Fbeta {val_fbeta}')
        print()
        history['train_fbeta'].append(train_fbeta)
        history['train_loss'].append(train_loss)
        history['val_fbeta'].append(val_fbeta)
        history['val_loss'].append(val_loss)
        if val_fbeta > best_fbeta:
            file_name = f"./models/model_gbertbase_{label}_{t}.bin"
            torch.save(model.state_dict(), file_name)
            for k,v in val_metrics.items():
                mlflow_logger.add_metric(k, v)
            for k,v in val_params.items():
                mlflow_logger.add_param(k, v)
            for k,v in train_metrics.items():
                mlflow_logger.add_metric(k, v)
            for k,v in train_params.items():
                mlflow_logger.add_param(k, v)
            best_fbeta = val_fbeta

    #############################################
    #MLflow logging

    constant_params = {
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'max_len': MAX_LEN,
                }
    mlflow_logger.add_tag("cycle4", True)
    mlflow_logger.add_param("normalization", 'norm')
    mlflow_logger.add_param("vectorizer", 'deepset/gbert-base')
    mlflow_logger.add_param("model", "deepset/gbert-base")
    # I'm re-using the grid_search_params field in order to no open too many mlflow columns
    mlflow_logger.add_param("grid_search_params", str(constant_params)[:249]) 
    mlflow_logger.add_param("lr", learning_rate)
    mlflow_logger.add_param('pos_weight', positive_class_weight)

    mlflow_logger.add_param("saved_model", f"model_gbertbase_{label}_{t}")
    mlflow_logger.add_param("label", data.current_label)
    mlflow_logger.add_param("balance_method", data.balance_method)
    if data.balance_method:
        mlflow_logger.add_param("sampling_strategy", data.sampling_strategy)
    mlflow_logger.add_model(None)

    with mlflow.start_run(run_name='deepset/gbert-base') as run:
        mlflow_logger.log()


# %%
def load_model_and_evaluate(state_dict: str, data: m.Posts, label:str, positive_class_weight:float):
    BATCH_SIZE = 8
    MAX_LEN = 264
    #EPOCHS = 10

    # ## MLflow instantiation
    IS_DEVELOPMENT = False
    mlflow_logger = m.MLFlowLogger(
        uri=TRACKING_URI,
        experiment=EXPERIMENT_NAME,
        is_dev=IS_DEVELOPMENT,
    )

    # ## Data loading
    normalize = lambda x: cleaning.normalize(x, url_emoji_dummy=False, pure_words=False)

    X_train, y_train = data.get_X_y('train')
    X_train = X_train.apply(normalize)
    df_train = pd.concat([X_train,y_train], axis=1)
    df_train.columns = ['text', label]

    X_val, y_val = data.get_X_y('val', balance_method='translate')
    X_val = X_val.apply(normalize)
    df_val = pd.concat([X_val,y_val], axis=1)
    df_val.columns = ['text', label]

    X_test, y_test = data.get_X_y('test', balance_method=None)
    X_test = X_test.apply(normalize)
    df_test = pd.concat([X_test,y_test], axis=1)
    df_test.columns = ['text', label]

    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    train_data_loader = create_data_loader(df_train, label, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, label, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, label, tokenizer, MAX_LEN, BATCH_SIZE)

    # ## Instantiation
    model = BinaryClassifier().to(device)
    #optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    mlflow_logger.add_param('optimizer', 'AdamW')
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    # From the BCEWithLogitLoss() documentation:
    # > For example, if a dataset contains 100 positive and 300 negative examples of a single class, 
    # > then pos_weight for the class should be equal to 300/100=3 . 
    # > The loss would act as if the dataset contains 3*100=300 positive examples.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([positive_class_weight])).to(device)

    # Loading the model
    model.load_state_dict(torch.load(f"./models/{state_dict}.bin"))
    model.eval()


    #history = defaultdict(list)
    train_fbeta, train_metrics, train_params,  train_loss = eval_model(
        model,
        train_data_loader,
        loss_fn,
        device,
        'train',
    )
    logger.info(f'Train loss {train_loss} Fbeta {train_fbeta}')
    val_fbeta, val_metrics, val_params,  val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        'val',
    )
    logger.info(f'Val   loss {val_loss} Fbeta {val_fbeta}')
    test_fbeta, test_metrics, test_params,  test_loss = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        'test',
    )
    logger.info(f'Test  loss {test_loss} Fbeta {test_fbeta}')
    print()
    #history['train_fbeta'].append(train_fbeta)
    #history['train_loss'].append(train_loss)
    #history['val_fbeta'].append(val_fbeta)
    #history['val_loss'].append(val_loss)
    for k,v in train_metrics.items():
        mlflow_logger.add_metric(k, v)
    for k,v in train_params.items():
        mlflow_logger.add_param(k, v)

    for k,v in val_metrics.items():
        mlflow_logger.add_metric(k, v)
    for k,v in val_params.items():
        mlflow_logger.add_param(k, v)

    for k,v in test_metrics.items():
        mlflow_logger.add_metric(k, v)
    for k,v in test_params.items():
        mlflow_logger.add_param(k, v)

    #############################################
    #MLflow logging

    #constant_params = {
    #                'epochs': EPOCHS,
    #                'batch_size': BATCH_SIZE,
    #                'max_len': MAX_LEN,
    #            }
    mlflow_logger.add_tag("cycle4", True)
    mlflow_logger.add_param("normalization", 'norm')
    mlflow_logger.add_param("vectorizer", 'deepset/gbert-base')
    mlflow_logger.add_param("model", "deepset/gbert-base")
    # I'm re-using the grid_search_params field in order to no open too many mlflow columns
    #mlflow_logger.add_param("grid_search_params", str(constant_params)[:249]) 
    #mlflow_logger.add_param("lr", learning_rate)
    mlflow_logger.add_param('pos_weight', positive_class_weight)

    mlflow_logger.add_param("saved_model", state_dict)
    mlflow_logger.add_param("label", data.current_label)
    mlflow_logger.add_param("balance_method", data.balance_method)
    if data.balance_method:
        mlflow_logger.add_param("sampling_strategy", data.sampling_strategy)
    mlflow_logger.add_model(None)

    with mlflow.start_run(run_name='deepset/gbert-base') as run:
        mlflow_logger.log()



# %%
#if __name__ == "__main__":
if False:
    TARGET_LABELS = ['label_discriminating', 'label_inappropriate',
        'label_sentimentnegative', 'label_needsmoderation']
    #TARGET_LABELS = ['label_needsmoderation']
    TRANS_OS = {'translate':[0.9], 'oversample':[0.9]}
    LEARNING_RATES = [1e-5, 5e-6, 25e-7, 125e-8]
    POSITIVE_WEIGHTS = [2., 3.]
    for positive_class_weight in POSITIVE_WEIGHTS:
        for learning_rate in LEARNING_RATES:
            for label in TARGET_LABELS:
                for method, strat in TRANS_OS.items():
                    for strategy in strat:
                        data = m.Posts()
                        data.set_label(label=label)
                        data.set_balance_method(balance_method=method, sampling_strategy=strategy)

                        logger.info('-' * 50)
                        logger.info(f'positive_class_weight: {positive_class_weight}, learning-rate: {learning_rate}')
                        logger.info(f'Label: {label}')
                        logger.info(f'Balance-method: {method}, Balance-strategy: {strategy}')
                        logger.info('-' * 50)

                        make_model(data, label, learning_rate, positive_class_weight)

if __name__ == "__main__":
    positive_class_weight = 2.0
    label = 'label_needsmoderation'

    logger.info(f'Label: {label}')
    data = m.Posts()
    data.set_label(label=label)
    data.set_balance_method(balance_method='translate', sampling_strategy=0.9)

    state_dict = 'model_gbertbase_label_needsmoderation_210423_014254'
    logger.info(f"Loading model from state_dict ./models/{state_dict}.bin")
    load_model_and_evaluate(state_dict,data, label, 2.0)

    