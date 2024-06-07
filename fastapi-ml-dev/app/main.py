from typing import Union
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load the best models for each target and soil type
soil_types = ['clay', 'sand', 'silt']
targets = ['lab_pH', 'lab_N', 'lab_P', 'lab_K', 'lab_EC']

# Define the model file mapping
model_files = {
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

# Load all models into a dictionary
models = {
    soil_type: {
        target: joblib.load(model_files[soil_type][target])
        for target in targets
    }
    for soil_type in soil_types
}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/add_sample", tags=["Training Section"])
def add_sample(var1: float, var2: float, var3: float, var4: float):
    return {"message": f"add {var1} {var2} {var3} {var4}"}

@app.post("/train", tags=["Training Section"])
def train():
    return {"message": f"Model trained successfully with RMSE: {4.332}"}

@app.post("/commit", tags=["Training Section"])
def commit():
    return {"message": f"Model has been updated"}

@app.get("/predict", tags=["Prediction Section"])
def predict(
    soil_type: str, 
    test_Temp: float, 
    test_Humid: float, 
    test_pH: float, 
    test_N: float, 
    test_P: float, 
    test_K: float, 
    test_Conductivity: float
):
    if soil_type not in models:
        raise HTTPException(status_code=400, detail="Invalid soil type")
    
    features = [test_Temp, test_Humid, test_pH, test_N, test_P, test_K, test_Conductivity]
    input_array = np.array([features])
    
    predictions = {target: models[soil_type][target].predict(input_array)[0] for target in targets}
    
    return predictions
