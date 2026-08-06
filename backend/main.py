import os
import json
import traceback
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "locations.json")
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load the trained machine learning model pipeline
try:
    model_pipeline = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
    """Serve the frontend HTML user interface"""
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>House Price Predictor API is Running!</h1>"

@app.get("/locations.json")
def get_locations():
    """Retrieve filtered and grouped locations for frontend dropdown"""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Other"]

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify pipeline deployment status"""
    return {
        "status": "healthy", 
        "model_loaded": model_pipeline is not None,
        "model_type": "GradientBoosting_Regressor_Pipeline"
    }

@app.post("/predict")
async def predict(request: Request):
    """Process incoming features payload and return model prediction"""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form_data = await request.form()
            data = dict(form_data)

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Parse request payload into inference dataframe
        df_input = pd.DataFrame([data])
        
        if model_pipeline is None:
            raise HTTPException(status_code=500, detail="Trained model is not loaded on the server.")

        # Predict using the original trained model pipeline
        pred_log = model_pipeline.predict(df_input)
        
        # Inverse transform log-scale predictions
        if isinstance(pred_log, np.ndarray):
            pred_price = np.expm1(pred_log)[0] if pred_log.ndim > 0 else np.expm1(pred_log)
        else:
            pred_price = float(pred_log)

        return {"prediction": float(pred_price)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    