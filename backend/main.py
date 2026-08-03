import json
import os
import traceback
import urllib.request
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Pipeline Model safely with Google Drive Auto-Download
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "house_price.pkl")
json_path = os.path.join(BASE_DIR, "locations.json")

GOOGLE_DRIVE_FILE_ID = "1C-7Qx7fRqw6N_54v67r3EK_UHY78TYz0"
download_url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

try:
    if not os.path.exists(model_path):
        print("Downloading model file from Google Drive...")
        urllib.request.urlretrieve(download_url, model_path)
        print("Model downloaded successfully!")

    model_pipeline = joblib.load(model_path)
    print("Model loaded into memory successfully!")
except Exception as e:
    print("LOAD ERROR:", traceback.format_exc())
    model_pipeline = None


# Frontend HTML Route
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
    """Serves the frontend HTML index page"""
    html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(BASE_DIR, "frontend", "index.html")

    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>House Price Predictor API is Running!</h1>"


# Locations API Route
@app.get("/locations.json")
def get_locations():
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Other"]


# Health Check Route
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


# Prediction Route
@app.post("/predict")
async def predict(request: Request):
    try:
        if model_pipeline is None:
            raise HTTPException(
                status_code=500,
                detail="Model file was not loaded correctly on server startup.",
            )

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form_data = await request.form()
            data = dict(form_data)

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        input_df = pd.DataFrame([data])
        
        prediction_log = model_pipeline.predict(input_df)
        prediction = float(np.expm1(prediction_log[0]))

        return {"prediction": prediction}

    except Exception as e:
        print(f"Prediction Error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))