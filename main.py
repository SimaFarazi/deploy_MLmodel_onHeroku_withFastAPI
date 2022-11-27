# Put the code for your API here.
from fastapi import FastAPI
from model import inference
from data import process_data
import joblib
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.encoders import jsonable_encoder


class Inputdf(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str= Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str= Field(alias="native-country")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/lb.joblib")
model = joblib.load('model/random_forest_model.pkl')

app = FastAPI()

@app.get("/")
async def create_greeting_msg():
    return {"greetings": "Welcome to salary predictor app!"}

@app.post("/inference/")
async def perform_model_inference(X:Inputdf):
    X = jsonable_encoder(X)
    X_df = pd.DataFrame(data = X, index=[0])

    X_test, y_test, encoder_, lb_ = process_data(
    X_df, categorical_features=cat_features, label=None, training=False, 
    encoder=encoder, lb=lb
    )
    
    prediction = inference(model, X_test)

    """
    https://stackoverflow.com/questions/69543228/trouble-fixing-cannot-convert-dictionary-update-sequence-element-0-to-a-seque
    ut FastApi use jsonable_encoder in a response,
    and numpy array is not acceptable. 
    You should convert to list(prediction.tolist()).
    """
    return{"prediction":prediction.tolist()}