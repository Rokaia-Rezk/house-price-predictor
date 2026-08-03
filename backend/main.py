import os
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import os
# 1. تحديث الأعمدة لتطابق اللي اتدرب عليه الموديل بالظبط
class PredictionRequest(BaseModel):
    carpet_area_sqft: float = Field(..., gt=0, description="Area in sqft")
    bedrooms: int = Field(1, ge=1)
    bathroom: int = Field(1, ge=1)
    balcony: int = Field(0, ge=0)
    floor_num: int = Field(0, ge=0)
    total_floors: int = Field(1, ge=1)
    location_grouped: str = Field("Other", description="Location name")
    Furnishing: str = Field("Unfurnished", description="Furnishing status")
    facing: str = Field("North", description="Facing direction")
    Transaction: str = Field("Resale", description="Transaction type")
    Ownership: str = Field("Freehold", description="Ownership type")

model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "house_price.pkl")
        model_pipeline = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    yield

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict_price(request: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # تحويل البيانات إلى DataFrame بنفس أسماء الأعمدة الدقيقة
    input_data = pd.DataFrame([{
        "carpet_area_sqft": request.carpet_area_sqft,
        "bedrooms": request.bedrooms,
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

    try:
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
    return """
    <html>
        <head>
            <title>House Price Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9; }
                h1 { color: #4A154B; }
                a { display: inline-block; margin-top: 20px; padding: 12px 24px; background-color: #6B46C1; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; }
                a:hover { background-color: #5A32A3; }
            </style>
        </head>
        <body>
            <h1>🚀 House Price Predictor API is Live!</h1>
            <p>Welcome to the End-to-End ML Prediction Service.</p>
            <a href="/docs">Open API Documentation (Swagger UI)</a>
        </body>
    </html>
    """