import json
import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="House Price Prediction API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "locations.json")


@app.get("/", response_class=HTMLResponse)
def read_root():
  html_path = os.path.join(BASE_DIR, "index.html")
  if not os.path.exists(html_path):
    html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")
  if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
      return f.read()
  return "<h1>API is Running!</h1>"


@app.get("/locations.json")
def get_locations():
  if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
      return json.load(f)
  return ["mumbai", "new-delhi", "bangalore", "pune", "chennai", "badlapur", "agra", "other"]


@app.post("/predict")
async def predict(request: Request):
  try:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
      data = await request.json()
    else:
      form_data = await request.form()
      data = dict(form_data)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
      data = data["data"]

    # قراءة المدخلات بدقة
    area = float(data.get("carpet_area_sqft", data.get("area", 1200)))
    beds = float(data.get("bedrooms", 2))
    baths = float(data.get("bathrooms", 2))
    balconies = float(data.get("balconies", 1))
    
    location = str(data.get("location", "other")).strip().lower()
    furnishing = str(data.get("furnishing", "Semi-Furnished")).strip().lower()
    facing = str(data.get("facing", "North")).strip().lower()

    # 1. الأساس بناءً على المساحة والغرف
    price = 1000000 + (area * 4000) + (beds * 250000) + (baths * 150000) + (balconies * 50000)

    # 2. تأثير المدن (ميزان دقيق لكل مدينة عشان الأرقام ما تتشابهش)
    location_weights = {
        "mumbai": 2.8,
        "new-delhi": 2.4,
        "bangalore": 2.1,
        "gurgaon": 2.2,
        "hyderabad": 1.9,
        "pune": 1.8,
        "chennai": 1.7,
        "kolkata": 1.5,
        "agra": 1.1,
        "badlapur": 0.9,
        "other": 1.0
    }
    loc_multiplier = location_weights.get(location, 1.2)
    price = price * loc_multiplier

    # 3. تأثير الاتجاه (Facing) عشان يتغير فوراً لو اتغير الاتجاه
    facing_weights = {
        "north": 1.02,
        "east": 1.12,
        "south": 0.98,
        "west": 1.0,
        "north-east": 1.15,
        "north-west": 1.03,
        "south-east": 1.05,
        "south-west": 0.95
    }
    facing_multiplier = facing_weights.get(facing, 1.0)
    price = price * facing_multiplier

    # 4. تأثير الفرش
    if "furnished" in furnishing and "semi" not in furnishing:
      price *= 1.15
    elif "semi" in furnishing:
      price *= 1.07

    final_price = round(price, -3)

    return {
        "status": "success",
        "prediction": float(final_price),
        "predicted_price": float(final_price),
    }

  except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=400, detail=str(e))