from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import joblib
import sys
path_root = Path(__file__).parent
sys.path.append(str(path_root))

from starter.ml.model import inference  # nopep8
from starter.ml.data import process_data  # nopep8

# Define the model schema


class Input(BaseModel):
    age: int = 45
    capital_gain: int = Field(2174, alias="capital-gain")
    capital_loss: int = Field(0, alias="capital-loss")
    education: str = "Bachelors"
    education_num: int = Field(13, alias="education-num")
    fnlgt: int = 2334
    hours_per_week: int = Field(60, alias="hours-per-week")
    marital_status: str = Field("Never-married", alias="marital-status")
    native_country: str = Field("Cuba", alias="native-country")
    occupation: str = "Prof-specialty"
    race: str = "Black"
    relationship: str = "Wife"
    sex: str = "Female"
    workclass: str = "State-gov"

    class Config:
        allow_population_by_field_name = True


# Instantiate the app.
app = FastAPI()

# Load model and encoders
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")
model = joblib.load("model/model.pkl")


@app.get("/")
async def say_hello():
    return {"greeting": "This is an example of an scalable ml pipeline!"}


@app.post("/predict")
def predict(input: Input):
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
    X, _, _, label_encoder = process_data(
        pd.DataFrame(input.dict(by_alias=True), index=[0]),
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb)
    # Make the prediction
    prediction = inference(model, X)
    # Return the prediction
    return {"prediction": label_encoder.inverse_transform(prediction)[0]}
