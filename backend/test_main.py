from fastapi.testclient import TestClient
from backend.main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "carpet_area_sqft": 1200.0,
        "bedrooms": 2,
        "bathroom": 2,
        "balcony": 1,
        "floor_num": 2,
        "total_floors": 5,
        "location_grouped": "Other",
        "Furnishing": "Semi-Furnished",
        "facing": "North",
        "Transaction": "Resale",
        "Ownership": "Freehold"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()