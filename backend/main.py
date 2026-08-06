import json
import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="House Price Prediction API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "locations.json")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def read_root():
  html_path = os.path.join(BASE_DIR, "index.html")
  if not os.path.exists(html_path):
    html_path = os.path.join(BASE_DIR, "..", "frontend", "index.html")
  if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
      return f.read()
  return "<h1>House Price Predictor API is Running!</h1>"


@app.get("/locations.json", tags=["Metadata"])
def get_locations():
  if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
      return json.load(f)
  return [
      "agra", "ahmedabad", "allahabad", "aurangabad", "badlapur", "bangalore", 
      "bhiwadi", "bhubaneswar", "chandigarh", "chennai", "coimbatore", "dehradun", 
      "faridabad", "ghaziabad", "goa", "greater-noida", "guntur", "gurgaon", 
      "guwahati", "hyderabad", "jaipur", "jamshedpur", "kalyan", "kanpur", 
      "kochi", "kolkata", "lucknow", "mangalore", "mohali", "mumbai", "nagpur", 
      "nashik", "navi-mumbai", "new-delhi", "noida", "other", "palghar", 
      "panchkula", "patna", "pune", "raipur", "ranchi", "siliguri", "sonipat", 
      "surat", "thane", "vadodara", "varanasi", "vijayawada", "visakhapatnam", "zirakpur"
  ]


@app.get("/health", tags=["Health"])
def health_check():
  return {"status": "healthy", "engine": "Clean_Realistic_Pricing"}


@app.post("/predict", tags=["Prediction"])
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

    # Extract inputs safely
    area = float(data.get("carpet_area_sqft", data.get("area", 1200)))
    beds = float(data.get("bedrooms", 2))
    baths = float(data.get("bathrooms", 2))
    balconies = float(data.get("balconies", 1))
    
    location = str(data.get("location", "Other")).strip().lower()
    furnishing = str(data.get("furnishing", "Semi-Furnished")).strip().lower()
    facing = str(data.get("facing", "North")).strip().lower()
    transaction = str(data.get("transaction", "Resale")).strip().lower()
    ownership = str(data.get("ownership", "Freehold")).strip().lower()

    # Realistic base calculation for Indian real estate market scale
    base_price = 1500000 + (area * 3500) + (beds * 200000) + (baths * 100000) + (balconies * 50000)

    # City multipliers
    tier1 = ["mumbai", "new-delhi", "bangalore", "hyderabad", "chennai", "gurgaon", "pune", "kolkata", "navi-mumbai"]
    tier2 = ["agra", "ahmedabad", "jaipur", "lucknow", "chandigarh", "kochi", "surat", "bhubaneswar", "vadodara", "thane"]
    
    if location in tier1:
      loc_factor = 1.5
    elif location in tier2:
      loc_factor = 1.2
    else:
      loc_factor = 1.0

    # Modifiers
    furnish_factor = 1.15 if "furnished" in furnishing else (1.05 if "semi" in furnishing else 1.0)
    facing_factor = 1.08 if facing in ["east", "north-east"] else 1.0
    trans_factor = 1.1 if "new" in transaction else 1.0

    final_price = base_price * loc_factor * furnish_factor * facing_factor * trans_factor

    # Clean rounding to avoid weird decimal tails
    final_price = round(final_price, -3)
    final_price = max(800000.0, min(final_price, 50000000.0))

    return {
        "status": "success",
        "prediction": float(final_price),
        "predicted_price": float(final_price),
    }

  except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=400, detail=str(e))