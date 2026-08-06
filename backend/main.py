import os
import json
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

import os
import requests
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

import os
import requests
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(model_path):
  print("Downloading model from Hugging Face...")
  url = "https://huggingface.co/Rokaa2006/house-price-mode/raw/main/model.pkl"

  response = requests.get(url, stream=True)
  with open(model_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=32768):
      if chunk:
        f.write(chunk)
  print("Model downloaded successfully!")

try:
  model_pipeline = joblib.load(model_path)
  print("Model pipeline loaded successfully into memory!")
except Exception as e:
  print(f"Error loading model: {e}")
  model_pipeline = None


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
      "model_type": "Trained_RandomForest_Pipeline",
  }


@app.post("/predict")
async def predict(request: Request):
  try:
    if model_pipeline is None:
      raise HTTPException(
          status_code=500, detail="Model pipeline is not loaded on the server."
      )

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
      data = await request.json()
    else:
      form_data = await request.form()
      data = dict(form_data)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
      data = data["data"]

    # Convert request dictionary into a DataFrame matching training features
    input_df = pd.DataFrame([data])

    # Ensure numeric columns are properly converted
    numeric_cols = [
        "carpet_area_sqft",
        "floor_num",
        "bathroom",
        "balcony",
        "bedrooms",
    ]
    for col in numeric_cols:
      if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # Predict using the real pipeline (applying expm1 since training used log1p)
    log_pred = model_pipeline.predict(input_df)
    predicted_price = np.expm1(log_pred[0])

    return {
        "status": "success",
        "prediction": float(predicted_price),
        "predicted_price": round(float(predicted_price), 2),
    }

  except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=400, detail=str(e))