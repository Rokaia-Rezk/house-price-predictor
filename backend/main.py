import glob
import json
import os
import traceback
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

# Determine Absolute Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Search explicitly in current directory or app directory
possible_paths = [
    os.path.join(BASE_DIR, "house_price.pkl"),
    os.path.join(BASE_DIR, "house_price (2).pkl"),
    "house_price.pkl",
    "/app/house_price.pkl"
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

json_path = os.path.join(BASE_DIR, "locations.json")
if not os.path.exists(json_path):
    json_path = "locations.json"

model_pipeline = None
load_error_msg = ""

if model_path:
    try:
        model_pipeline = joblib.load(model_path)
        print(f"SUCCESS: Model loaded into memory from {model_path}")
    except Exception as e:
        load_error_msg = f"Joblib load failed: {str(e)}"
        print(f"LOAD ERROR at {model_path}:", traceback.format_exc())
else:
    load_error_msg = f"Model file not found. Checked: {possible_paths}"
    print("ERROR:", load_error_msg)


# Frontend HTML Route
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
    """Serves the frontend HTML index page"""
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")

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
    return {
        "status": "healthy", 
        "model_loaded": model_pipeline is not None,
        "model_path_used": model_path,
        "error_details": load_error_msg
    }


# Prediction Route
@app.post("/predict")
async def predict(request: Request):
    try:
        if model_pipeline is None:
            raise HTTPException(
                status_code=500,
                detail=f"Model error: {load_error_msg}",
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction Error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))