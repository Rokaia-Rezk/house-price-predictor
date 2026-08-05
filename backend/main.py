import os
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
model_path = os.path.join(BASE_DIR, "house_pricee.pkl")
json_path = os.path.join(BASE_DIR, "locations.json")

model_pipeline = None
load_error_msg = ""

try:
    # تحميل الموديل الحقيقي المدرب مباشرة بدون أي بدائل وهمية
    model_pipeline = joblib.load(model_path)
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    load_error_msg = str(e)
    print("ERROR:", load_error_msg)

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
            raise HTTPException(status_code=500, detail="Model not loaded: " + load_error_msg)

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form_data = await request.form()
            data = dict(form_data)


        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # تحويل البيانات القادمة إلى DataFrame عشان تدخل مظبوطة للـ Pipeline الحقيقي
        df_input = pd.DataFrame([data])
        
        # التنبؤ بالقيم باستخدام الموديل الحقيقي
        pred_log = model_pipeline.predict(df_input)
        pred_price = np.expm1(pred_log)[0]

        return {"prediction": float(pred_price)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))