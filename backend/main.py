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

@app.get("/", response_class=HTMLResponse)
def read_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "frontend", "index.html")
    
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend file not found</h1>"