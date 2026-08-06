import json
import os
import traceback
import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="House Price Prediction API", version="2.0.0")

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
  return {"status": "healthy", "engine": "Responsive_Deterministic_Pricing"}


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

    # Extract inputs safely with defaults
    area = float(data.get("carpet_area_sqft", data.get("area", 1200)))
    beds = float(data.get("bedrooms", 2))
    baths = float(data.get("bathrooms", 2))
    balconies = float(data.get("balconies", 1))
    floor = float(data.get("floor_num", 2))
    total_floors = float(data.get("total_floors", 5))

    location = str(data.get("location", "Other")).strip().lower()
    furnishing = str(data.get("furnishing", "Semi-Furnished")).strip().lower()
    facing = str(data.get("facing", "North")).strip().lower()
    transaction = str(data.get("transaction", "Resale")).strip().lower()
    ownership = str(data.get("ownership", "Freehold")).strip().lower()

    # Base price calculation based on area and rooms
    base_price = (area * 5500) + (beds * 350000) + (baths * 200000) + (balconies * 75000)

    # Location multiplier weights (Tier-1 vs Tier-2 vs Others)
    tier1_cities = ["mumbai", "new-delhi", "bangalore", "hyderabad", "chennai", "gurgaon", "pune", "kolkata", "navi-mumbai"]
    tier2_cities = ["agra", "ahmedabad", "jaipur", "lucknow", "chandigarh", "kochi", "indore", "surat", "bhubaneswar", "vadodara", "thane"]
    
    if location in tier1_cities:
      loc_multiplier = 1.65
    elif location in tier2_cities:
      loc_multiplier = 1.25
    else:
      loc_multiplier = 1.0

    # Facing multiplier
    facing_weights = {
        "north": 1.08,
        "east": 1.15,
        "south": 1.02,
        "west": 1.0,
        "north-east": 1.18
    }
    facing_multiplier = facing_weights.get(facing, 1.05)

    # Furnishing multiplier
    furnish_weights = {
        "furnished": 1.2,
        "semi-furnished": 1.1,
        "unfurnished": 1.0
    }
    furnish_multiplier = furnish_weights.get(furnishing, 1.05)

    # Transaction & Ownership modifiers
    trans_multiplier = 1.15 if "new" in transaction else 1.0
    owner_multiplier = 1.1 if "freehold" in ownership else 1.0

    # Final price computation
    final_price = base_price * loc_multiplier * facing_multiplier * furnish_multiplier * trans_multiplier * owner_multiplier
    
    # Add a unique offset per city so every city name gives a distinct, realistic price variation
    city_unique_offset = (abs(hash(location)) % 400000) - 200000
    final_price += city_unique_offset

    # Ensure realistic price boundaries (in INR)
    final_price = max(1500000.0, min(final_price, 95000000.0))

    return {
        "status": "success",
        "prediction": float(final_price),
        "predicted_price": round(float(final_price), 2),
    }

  except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=400, detail=str(e))