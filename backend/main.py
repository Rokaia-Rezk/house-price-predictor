import os
import joblib
import json
import traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
model_path = os.path.join(BASE_DIR, "model.pkl")
json_path = os.path.join(BASE_DIR, "locations.json")
csv_path = os.path.join(BASE_DIR, "..", "notebooks", "house_prices.csv") # لو الـ csv موجود

model_pipeline = None
load_error_msg = ""

try:
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
        model_pipeline = joblib.load(model_path)
        print("SUCCESS: Model loaded from file!")
    else:
        raise Exception("Model file is missing or too small (LFS pointer issue)")
except Exception as e:
    print("WARNING: Training fallback model on the fly to guarantee 100% working API...")
    try:
        # لو ملف الـ CSV موجود جنبه، ندربه فوراً ونخلص!
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, "house_prices.csv")
            
        df = pd.read_csv(csv_path)
        # تنظيف سريع وتدريب نموذج سريع جداً يشتغل فوراً
        if 'price' in df.columns:
            y = np.log1p(df['price'])
            X = df.drop(columns=['price'])
            
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
            ])
            model_pipeline.fit(X, y)
            joblib.dump(model_pipeline, model_path)
            print("SUCCESS: Fallback model trained and saved successfully on the fly!")
    except Exception as ex:
        load_error_msg = str(ex)
        print("ERROR in fallback training:", load_error_msg)

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

        df_input = pd.DataFrame([data])
        pred_log = model_pipeline.predict(df_input)
        pred_price = np.expm1(pred_log)[0]

        return {"prediction": float(pred_price)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))