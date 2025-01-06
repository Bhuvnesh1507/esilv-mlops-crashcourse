from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_VERSION,
    PATH_TO_MODEL,
    PATH_TO_PREPROCESSOR,
)
from fastapi import FastAPI
 
from lib.modelling import run_inference
from lib.models import WineData, PredictionOut
from lib.utils import load_model, load_preprocessor
 
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)
 
 
@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION}
 
 
@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: WineData):
    scaler = load_preprocessor(PATH_TO_PREPROCESSOR)
    model = load_model(PATH_TO_MODEL)
    prediction = run_inference([payload], scaler, model)
    return {"quality_prediction": prediction[0]}