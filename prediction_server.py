from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

MODELNAME = "NaiveBayes_label_sentimentnegative_2021-04-08_200950"
MODELNAME_NEEDSMODERATION = "SupportVectorMachine_label_negative_2021-04-22_070830"
MODELNAME_INAPPROPRIATE = ""
MODELNAME_DISCRIMINATING = ""
MODELNAME_SENTIMENTNEGATIVE = "RandomForest_label_sentimentnegative_2021-04-21_185343"
PATH = Path("models")

model_needsmoderation = pickle.load(open(PATH / MODELNAME_NEEDSMODERATION / 'model.pkl', 'rb'))
model_sentimentnegative = pickle.load(open(PATH / MODELNAME_SENTIMENTNEGATIVE / 'model.pkl', 'rb'))

app = FastAPI()

class Post(BaseModel):
    text: str


def predict_needsmoderation(text):
    prediction = model_needsmoderation.predict(text)
    return prediction[0]


def predict_sentiment_negative(text):
    prediction = model_sentimentnegative.predict_proba(text)
    return prediction[0][1]


@app.post('/predict')
async def predict_post(post: Post):
    data = post.dict()
    data_in = pd.Series(data['text'])
    needs_moderation = predict_needsmoderation(data_in)
    response = {'needsmoderation': needs_moderation, 'sentimentnegative': 0.0, 'inappropriate': 0.0, 'discriminating': 0.0}
    if needs_moderation:
        response["sentimentnegative"] = predict_sentiment_negative(data_in)
    return response