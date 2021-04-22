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
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from datetime import datetime

from utils import loading, feature_engineering, augmenting

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
def create_data_loader(df, label, tokenizer, max_len, batch_size):
    ds = OMPDataset(
        text=df.text.to_numpy(),
        targets=df[label].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


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
        return torch.sigmoid(output)

# %%
def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
    model = model.train()
    losses = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        preds = torch.round(outputs)
        targets = targets.unsqueeze(1)
        targets = targets.float()
        loss = loss_fn(outputs, targets)
        tp += (targets * preds).sum(dim=0).to(torch.float32)
        tn += ((1 - targets) * (1 - preds)).sum(dim=0).to(torch.float32)
        fp += ((1 - targets) * preds).sum(dim=0).to(torch.float32)
        fn += (targets * (1 - preds)).sum(dim=0).to(torch.float32)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        div = tp + 0.5 * ( fp + fn)
    return tp / div if div != 0 else 0, np.mean(losses)


# %%
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            targets = targets.unsqueeze(1)
            targets = targets.float()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
            preds = torch.round(outputs)
            loss = loss_fn(outputs, targets)
            tp += (targets * preds).sum(dim=0).to(torch.float32)
            tn += ((1 - targets) * (1 - preds)).sum(dim=0).to(torch.float32)
            fp += ((1 - targets) * preds).sum(dim=0).to(torch.float32)
            fn += (targets * (1 - preds)).sum(dim=0).to(torch.float32)
            losses.append(loss.item())
            div = tp + 0.5 * ( fp + fn)
    return tp / div if div != 0 else 0, np.mean(losses)


# %% [markdown]
# ## Main method

# %%
def make_model(label):
    BATCH_SIZE = 2
    MAX_LEN = 264
    EPOCHS = 10
    LEARNING_RATE = 1e-5

    # ## Data loading
    df_train = loading.load_extended_posts(split='train', label=label)
    df_train = feature_engineering.add_column_text(df_train)
    X,y = augmenting.get_augmented_X_y(df_train.text, 
                                    df_train[label], 
                                    sampling_strategy=1, 
                                    label='label_sentimentnegative',
                                    )
    df_train = pd.concat([X,y], axis=1)
    df_val = loading.load_extended_posts(split='val', label=label)
    df_val = feature_engineering.add_column_text(df_val)

    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    train_data_loader = create_data_loader(df_train, label, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, label, tokenizer, MAX_LEN, BATCH_SIZE)
    #test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # ## Instantiation
    model = BinaryClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
        )
    loss_fn = nn.BCELoss().to(device)

    history = defaultdict(list)
    best_f1 = 0
    t = datetime.now().strftime("%y%m%d_%H%M")
    for epoch in range(EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}')
        logger.info('-' * 10)
        train_f1, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        logger.info(f'Train loss {train_loss} F1 {train_f1}')
        val_f1, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        logger.info(f'Val   loss {val_loss} F1 {val_f1}')
        print()
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)
        if val_f1 > best_f1:
            file_name = f"./models/model_gbert_pool_{label}_{t}.bin"
            torch.save(model.state_dict(), file_name)
            best_f1 = val_f1

# %%
if __name__ == "__main__":
    LABELS = ['label_sentimentnegative']
    for l in LABELS:
        make_model(l)
    