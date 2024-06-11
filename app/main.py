from typing import Union
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import logging

app = FastAPI()

# Initialize the Service

admin_password = ""

# Define the model paths
model_paths = {
    'clay': {
        'lab_pH': 'RandomForestRegressor_clay_lab_pH.joblib',
        'lab_N': 'GradientBoostingRegressor_clay_lab_N.joblib',
        'lab_P': 'LinearRegression_clay_lab_P.joblib',
        'lab_K': 'RandomForestRegressor_clay_lab_K.joblib',
        'lab_EC': 'GradientBoostingRegressor_clay_lab_EC.joblib'
    },
    'sand': {
        'lab_pH': 'RandomForestRegressor_sand_lab_pH.joblib',
        'lab_N': 'GradientBoostingRegressor_sand_lab_N.joblib',
        'lab_P': 'GradientBoostingRegressor_sand_lab_P.joblib',
        'lab_K': 'RandomForestRegressor_sand_lab_K.joblib',
        'lab_EC': 'LinearRegression_sand_lab_EC.joblib'
    },
    'silt': {
        'lab_pH': 'GradientBoostingRegressor_silt_lab_pH.joblib',
        'lab_N': 'GradientBoostingRegressor_silt_lab_N.joblib',
        'lab_P': 'RandomForestRegressor_silt_lab_P.joblib',
        'lab_K': 'GradientBoostingRegressor_silt_lab_K.joblib',
        'lab_EC': 'GradientBoostingRegressor_silt_lab_EC.joblib'
    }
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all models into a dictionary
models = {}
try:
    for soil_type in model_paths:
        models[soil_type] = {}
        for target, path in model_paths[soil_type].items():
            model_path = os.path.join("/app/model", path)
            logger.info(f"Loading model for {soil_type} - {target} from {model_path}")
            models[soil_type][target] = joblib.load(model_path)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# Default Section ==============================================================

@app.get("/")
def read_root():
    # Return Report or status
    return {"Hello": "World"}

# Training Model Section ==============================================================

@app.post("/add_sample", tags=["Training Section"])
def add_sample(var1: float, var2: float, var3: float, var4: float):
    # Add samples to the training dataset
    return {"message": f"add {var1} {var2} {var3} {var4}"}

@app.post("/train", tags=["Training Section"])
def train():
    # Populate data and re fitting
    # Get performance metric (RMSE etc.)
    return {"message": f"Model trained successfully with RMSE: {4.332}"}

@app.post("/commit", tags=["Training Section"])
def commit():
    # Populate - Re-train Model - Save to file
    # Retrieve Model
    return {"message": f"Model has been updated"}

# Prediction Section ==============================================================

@app.get("/predict", tags=["Prediction Section"])
def predict(
    soil_type: str,
    temp: float,
    humid: float,
    ph: float,
    n: float,
    p: float,
    k: float,
    ec: float
):
    if soil_type not in models:
        raise HTTPException(status_code=400, detail="Invalid soil type")

    features = np.array([[temp, humid, ph, n, p, k, ec]])
    
    predictions = {}
    for target, model in models[soil_type].items():
        predictions[target] = model.predict(features)[0]

    return predictions
