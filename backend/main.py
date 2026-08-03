import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
model_path = os.path.join(current_dir, "house_price.pkl")
try:
    model_pipeline = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# Frontend HTML Route (هذه الدالة هي التي ترجع صفحة الموقع الواجهة)
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

# Health Check Route
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

# Prediction Route
@app.post("/predict")
def predict(data: dict):
    try:
        if model_pipeline is None:
            raise HTTPException(status_code=500, detail="Model file was not loaded correctly on server startup.")
        
        # تحويل البيانات لـ DataFrame والتوقع
        input_df = pd.DataFrame([data])
        prediction_log = model_pipeline.predict(input_df)
        prediction = float(np.exp(prediction_log[0])) # أو expm1 حسب تدريب الموديل

        return {"prediction": prediction}

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    