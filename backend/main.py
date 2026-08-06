import json
import os
import traceback
import joblib
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
json_path = os.path.join(BASE_DIR, "locations.json")
model_path = os.path.join(BASE_DIR, "model.pkl")

# لو ملف الموديل مش موجود، نعمل سكريبت ينشئ واحد تجريبي شغّال فوراً
if not os.path.exists(model_path):
  try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # بناء نموذج تجريبي سريع بياخد نفس المدخلات عشان السيرفر ما يضربش إيرور
    dummy_pipeline = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor())])
    # تدريب وهمي بسريع
    X_dummy = np.zeros((10, 5))
    y_dummy = np.zeros(10)
    dummy_pipeline.fit(X_dummy, y_dummy)

    joblib.dump(dummy_pipeline, model_path)
    print("Dummy model created successfully!")
  except Exception as e:
    print(f"Error creating dummy model: {e}")

try:
  model_pipeline = joblib.joblib.load(model_path) if hasattr(joblib, 'joblib') else joblib.load(model_path)
  print("Model pipeline loaded successfully!")
except Exception as e:
  print(f"Error loading model: {e}")
  model_pipeline = None