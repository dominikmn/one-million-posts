from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

from utils import feature_engineering

MODELNAME = "NaiveBayes_label_sentimentnegative_2021-04-08_200950"
PATH = Path("models")

app = FastAPI()

class Post(BaseModel):
    text: str

@app.post('/predict')
async def predict_post(post: Post):
    data = post.dict()
    loaded_model = pickle.load(open(PATH / MODELNAME / 'model.pkl', 'rb'))
    data_in = pd.Series(data['text'])
    prediction = loaded_model.predict(data_in)
    probability = loaded_model.predict_proba(data_in).max()
    return {'prediction': prediction[0],'probability': probability}