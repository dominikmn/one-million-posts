from transformers import BertModel, BertTokenizer
import torch

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import cleaning

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def create_data_loader(df:pd.DataFrame, label:str, tokenizer:BertTokenizer, max_len:int, batch_size:int) -> DataLoader:
    ds = OMPDataset(
        text=df.text.to_numpy(),
        targets=df[label].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )
    return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        ).values()
        output = self.dropout(pooled_output)
        output = self.out(output)
        return output

def get_model(state_dict_file: str) -> BinaryClassifier:
    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(f"{state_dict_file}"))
    return model


def get_prediction(text_list: list, model: BinaryClassifier) -> list:
    BATCH_SIZE = 1
    MAX_LEN = 264

    model.eval()

    # ## Data loading
    normalize = lambda x: cleaning.normalize(x, url_emoji_dummy=False, pure_words=False)

    X = pd.Series(text_list)
    X = X.apply(normalize)
    y = pd.Series([0]*len(text_list)) # dummy targets
    df = pd.concat([X,y], axis=1)
    df.columns = ['text', 'target']
    data_loader = create_data_loader(df, 'target', model.tokenizer, MAX_LEN, BATCH_SIZE)
    
    prediction_list = []
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
            preds = torch.sigmoid(outputs)
            prediction_list += [p[0] for p in preds.cpu().detach().numpy().tolist()]
    
    return prediction_list



