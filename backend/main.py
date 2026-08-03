from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

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


@app.post("/predict")
def predict(data: dict):
    try:
        # 1. التحرّك في حالة عدم وجود الموديل
        if model_pipeline is None:
            return {"error": "Model file was not loaded correctly on server startup."}, 500

        # 2. طباعة البيانات للـ Logs عشان نشوف الـ Frontend باعت إيه
        print("Received payload:", data)

        # 3. تحويل الـ JSON لـ DataFrame
        input_df = pd.DataFrame([data])

        # 4. التوقع
        prediction_log = model_pipeline.predict(input_df)
        
        # لو عاملة log transform للسعر (أو عدليها حسب الكود عندك)
        prediction = float(np.exp(prediction_log[0]))

        return {"prediction": prediction}

    except Exception as e:
        # هيرجع الخطأ التفصيلي في الـ Response بدل ما يضرب 500 صامت!
        print(f"Prediction Error: {str(e)}")
        return {"error": str(e)}, 400