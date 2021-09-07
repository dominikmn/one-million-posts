from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

from utils import gbert_models

MODELNAME = "NaiveBayes_label_sentimentnegative_2021-04-08_200950"
MODELNAME_NEEDSMODERATION = "models/model_gbertbase_label_needsmoderation_210423_014254.bin"
MODELNAME_INAPPROPRIATE = "models/model_gbertbase_label_inappropriate_210423_030629.bin"
MODELNAME_DISCRIMINATING = "models/model_gbertbase_label_discriminating_210423_025622.bin"
MODELNAME_SENTIMENTNEGATIVE = "models/model_gbertbase_label_sentimentnegative_210423_021224.bin"

model_needsmoderation = gbert_models.get_model(MODELNAME_NEEDSMODERATION)
model_inappropriate = gbert_models.get_model(MODELNAME_INAPPROPRIATE)
model_discriminating = gbert_models.get_model(MODELNAME_DISCRIMINATING)
model_sentimentnegative = gbert_models.get_model(MODELNAME_SENTIMENTNEGATIVE)

app = FastAPI()

class Post(BaseModel):
    text: str


def predict_needsmoderation(text):
    prediction = gbert_models.get_prediction([text], model_needsmoderation)
    return int(prediction[0] > 0.5)


def predict_inappropriate(text):
    prediction = gbert_models.get_prediction([text], model_inappropriate)
    return prediction[0]

def predict_discriminating(text):
    prediction = gbert_models.get_prediction([text], model_discriminating)
    return prediction[0]

def predict_sentimentnegative(text):
    prediction = gbert_models.get_prediction([text], model_sentimentnegative)
    return prediction[0]


@app.post('/predict')
async def predict_post(post: Post):
    text = post.dict()["text"]
    needs_moderation = predict_needsmoderation(text)
    response = {'needsmoderation': needs_moderation, 'sentimentnegative': 0.0, 'inappropriate': 0.0, 'discriminating': 0.0}
    if needs_moderation:
        response["sentimentnegative"] = predict_sentimentnegative(text)
        response["inappropriate"] = predict_inappropriate(text)
        response["discriminating"] = predict_discriminating(text)
    return response