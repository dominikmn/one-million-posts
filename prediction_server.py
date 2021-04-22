from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

from utils import feature_engineering

MODELNAME ="SupportVectorMachine_label_negative_2021-04-22_071002"
MODELNAME = "NaiveBayes_label_sentimentnegative_2021-04-08_200950"

PATH = Path("models")

app = FastAPI()

class Post(BaseModel):
    text: str


def predict_needsmoderation(text):
    loaded_model = pickle.load(open(PATH / MODELNAME / 'model.pkl', 'rb'))
    prediction = loaded_model.predict(text)
    return prediction[0]


@app.post('/predict')
async def predict_post(post: Post):
    data = post.dict()
    data_in = pd.Series(data['text'])
    needs_moderation = predict_needsmoderation(data_in)
    return {'needsmoderation': needs_moderation, 'sentimentnegative': 0.0, 'inappropriate': 0.0, 'discriminating': 0.0}