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

# %%
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
# %matplotlib inline
# %config InlineBackend.figure_format='retina'


# %% tags=[]
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

#model = AutoModelForMaskedLM.from_pretrained("deepset/gbert-base")
model = BertModel.from_pretrained("deepset/gbert-base")

# %%
type(model)

# %%
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

# %%
print(device)

# %% [markdown]
# ## Data loading

# %%
from utils import loading, feature_engineering, augmenting

# %%
LABEL = 'label_sentimentnegative'

# %%
df_train = loading.load_extended_posts(split='train', label=LABEL)
df_train = feature_engineering.add_column_text(df_train)

# %%
X,y = augmenting.get_augmented_X_y(df_train.text, 
                                   df_train[LABEL], 
                                   sampling_strategy=1, 
                                   label='label_sentimentnegative',
                                  )

# %%
df_train = pd.concat([X,y], axis=1)

# %%

# %%
tokenizer.unk_token, tokenizer.unk_token_id


# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# sample_txt = 'Wann war ich das letzte mal draußen? Ich bin schon seit zwei Wochen zuhause.'
#

# %% [markdown]
# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')

# %% [markdown]
# encoding = tokenizer.encode_plus(
#     sample_txt,
#     max_length=32,
#     add_special_tokens=True, # Add '[CLS]' and '[SEP]'
#     return_token_type_ids=False,
#     pad_to_max_length=True,
#     return_attention_mask=True,
#     return_tensors='pt',  # Return PyTorch tensors
# )
# encoding.keys()

# %%

# %%

# %%
#token_lens = []
#for txt in df_train.text:
#    tokens = tokenizer.encode(txt)
#    token_lens.append(len(tokens))

# %%
#max(token_lens)

# %%
#sns.distplot(token_lens)
#plt.xlim([0, 300]);
#plt.xlabel('Token count');

# %%

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
df_val = loading.load_extended_posts(split='val', label='label_sentimentnegative')
df_val = feature_engineering.add_column_text(df_val)


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
BATCH_SIZE = 4
MAX_LEN = 264
train_data_loader = create_data_loader(df_train, LABEL, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, LABEL, tokenizer, MAX_LEN, BATCH_SIZE)
#test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# %%
data = next(iter(train_data_loader))
data.keys()


# %% [markdown]
# print(data['input_ids'].shape)
# print(data['attention_mask'].shape)
# print(data['targets'].shape)
# torch.Size([16, 160])
# torch.Size([16, 160])
# torch.Size([16])

# %% [markdown]
# last_hidden_state, pooled_output = model(
#   input_ids=encoding['input_ids'],
#   attention_mask=encoding['attention_mask']
# ).values()

# %% [markdown]
# last_hidden_state

# %%
#print(last_hidden_state.shape)

# %% [markdown]
# print(model.config.hidden_size)
# print(model.config.vocab_size)

# %%
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        ).values()
        output = self.drop(pooled_output)
        output = self.out(output)
        return torch.sigmoid(output)


# %%
model = SentimentClassifier()
model = model.to(device)

# %%
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

# %% tags=[]
#nn.functional.softmax(model(input_ids, attention_mask), dim=1)

# %%
#torch.cuda.empty_cache()

# %%
#torch.cuda.memory_summary(device=None, abbreviated=False)

# %% [markdown]
# ## Training

# %%
EPOCHS = 10
LEARNING_RATE = 1e-5
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.BCELoss().to(device)


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


# %%
# %%time
history = defaultdict(list)
best_f1 = 0
t = datetime.now().strftime("%Y-%m-%d_%H%M")
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_f1, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print(f'Train loss {train_loss} F1 {train_f1}')
    val_f1, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f'Val   loss {val_loss} F1 {val_f1}')
    print()
    history['train_f1'].append(train_f1)
    history['train_loss'].append(train_loss)
    history['val_f1'].append(val_f1)
    history['val_loss'].append(val_loss)
    if val_f1 > best_f1:
        s = f"{val_f1:.2f}".replace('.','' 
        torch.save(model.state_dict(), f'./models/model_gbert_pool_{LABEL}_{t}_f1{s}.bin')
        best_f1 = val_f1

# %% [markdown]
# plt.plot(history['train_f1'], label='train F1')
# plt.plot(history['val_f1'], label='validation F1')
# plt.title('Training history')
# plt.ylabel('F1')
# plt.xlabel('Epoch')
# plt.legend()
# plt.ylim([0, 1]);

# %%
