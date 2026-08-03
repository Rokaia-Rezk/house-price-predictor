from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Pipeline Model
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "best_pipeline_model.pkl")
        if os.path.exists(model_path):
            model_pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")

# Request Schema
class PredictionRequest(BaseModel):
    BHK: float
    size_sqft: float
    bathroom: float
    balcony: float
    floor_num: float
    total_floors: float
    location_grouped: str
    Furnishing: str
    facing: str
    Transaction: str
    Ownership: str

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.post("/predict", tags=["Prediction"])
def predict_price(request: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    try:
        import pandas as pd
        input_data = pd.DataFrame([{
            "BHK": request.BHK,
            "size_sqft": request.size_sqft,
            "bathroom": request.bathroom,
            "balcony": request.balcony,
            "floor_num": request.floor_num,
            "total_floors": request.total_floors,
            "location_grouped": request.location_grouped,
            "Furnishing": request.Furnishing,
            "facing": request.facing,
            "Transaction": request.Transaction,
            "Ownership": request.Ownership
        }])
        
        log_price = model_pipeline.predict(input_data)[0]
        predicted_price = float(np.expm1(log_price))
        
        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "INR"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "frontend", "index.html")
    
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>House Price Predictor API is Running!</h1><p>Go to <a href='/docs'>/docs</a> for API Documentation.</p>"