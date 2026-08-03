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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_pipeline_model.pkl")

try:
    model_pipeline = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
    """Serves the frontend HTML index page"""
    html_path = os.path.join(current_dir, "..", "frontend", "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(current_dir, "frontend", "index.html")
        
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>House Price Predictor API is Running!</h1>"


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict", tags=["Prediction"])
def predict_price(payload: dict):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    try:
        import pandas as pd
        
        # Smart Dynamic Field Mapping for Frontend & Backend compatibility
        bhk_val = payload.get("BHK") or payload.get("bhk") or payload.get("bedrooms") or 0
        size_val = payload.get("size_sqft") or payload.get("carpet_area_sqft") or payload.get("carpet_area") or 0
        bath_val = payload.get("bathroom") or payload.get("bathrooms") or 0
        balcony_val = payload.get("balcony") or payload.get("balconies") or 0
        floor_val = payload.get("floor_num") or payload.get("floor") or 0
        total_floors_val = payload.get("total_floors") or 0
        
        data = {
            "BHK": float(bhk_val),
            "size_sqft": float(size_val),
            "bathroom": float(bath_val),
            "balcony": float(balcony_val),
            "floor_num": float(floor_val),
            "total_floors": float(total_floors_val),
            "location_grouped": str(payload.get("location_grouped") or payload.get("location") or "Other"),
            "Furnishing": str(payload.get("Furnishing") or payload.get("furnishing") or "Unfurnished"),
            "facing": str(payload.get("facing") or "North"),
            "Transaction": str(payload.get("Transaction") or payload.get("transaction") or "Resale"),
            "Ownership": str(payload.get("Ownership") or payload.get("ownership") or "Freehold")
        }

        input_data = pd.DataFrame([data])
        
        log_price = model_pipeline.predict(input_data)[0]
        predicted_price = float(np.expm1(log_price))
        
        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "INR"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")