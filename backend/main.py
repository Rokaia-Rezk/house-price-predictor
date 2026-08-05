import os
import requests
import joblib
import json
import traceback
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
model_path = os.path.join(BASE_DIR, "house_price.pkl")
json_path = os.path.join(BASE_DIR, "locations.json")

GITHUB_MODEL_URL = "https://github.com/Rokaia-Rezk/house-price-predictor/releases/download/v1.0.0/house_price.pkl"

model_pipeline = None
load_error_msg = ""

# تحميل ذكي وآمن جداً: لو الملف مش موجود، بنحاول ننزله في الخلفية بطريقة لا تؤثر على قيام السيرفر
def download_model_if_needed():
    global model_pipeline, load_error_msg
    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1024:
            print("Downloading model from GitHub Releases (Background)...")
            response = requests.get(GITHUB_MODEL_URL, timeout=300, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                print("Model downloaded successfully!")
        
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
            model_pipeline = joblib.load(model_path)
            print("SUCCESS: Model loaded into memory!")
        else:
            load_error_msg = "Model file still missing or empty."
    except Exception as e:
        load_error_msg = f"Download/Load error: {str(e)}"
        print("ERROR:", load_error_msg)

# تشغيل التحميل مرة واحدة
download_model_if_needed()

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>House Price Predictor API is Running!</h1>"

@app.get("/locations.json")
def get_locations():
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Other"]

@app.get("/health", tags=["Health"])
def health_check():
    # لو الموديل لسه محملهوش، نحاول تريك تحميلة تانية سريعة
    global model_pipeline
    if model_pipeline is None:
        download_model_if_needed()
        
    return {
        "status": "healthy", 
        "model_loaded": model_pipeline is not None,
        "model_path_used": model_path,
        "error_details": load_error_msg
    }

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