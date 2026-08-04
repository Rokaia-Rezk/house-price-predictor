# 🏡 Real Estate Price Predictor (End-to-End ML Web Application)

A robust, production-ready, end-to-end Machine Learning web application designed to predict real estate prices in India. This project covers the complete machine learning lifecycle: from raw data ingestion, meticulous Exploratory Data Analysis (EDA), advanced data cleaning and feature engineering, multi-model training and evaluation, to serving via a **FastAPI** backend and interacting through a sleek **React + TypeScript + Vite** frontend.

---

## 🚀 Project Overview & Journey

Building this project involved tackling messy real-world data head-on:
1. **Data Ingestion & Inspection:** Loaded and analyzed a massive dataset of ~187,000 real estate listings (`house_prices.csv`) sourced from Kaggle.
2. **Exploratory Data Analysis (EDA):** Generated visual insights including target price distributions (using Log-Scale), scatter plots of price vs. carpet area, bar charts for top locations, and box plots for furnishing status.
3. **Data Cleaning & Feature Engineering:** 
   * Parsed complex text strings in "Amount" (converting Lacs and Crores to exact numerical values) and "Carpet Area" (normalizing sqft and sqm).
   * Handled complex floor strings ("3 out of 10", "Ground", "Basement").
   * Grouped high-cardinality categorical features (`location` and `society`) to retain only the top-50 locations and grouped the rest into `"other"` to prevent curse of dimensionality.
   * Dropped useless/redundant columns and removed extreme outliers based on price-per-sqft (1st and 99th percentiles).
4. **Modeling & Pipeline Integration:** Built a scikit-learn pipeline bundling a `ColumnTransformer` (imputation, scaling, and one-hot encoding) directly with a **Random Forest Regressor** (outperforming the Linear Regression baseline).
5. **Backend & Frontend Implementation:** Developed a FastAPI backend with integrated CORS, lifespan model loading, and automated testing, paired with a fully validated React + TypeScript + Vite user interface.
6. **Deployment & Hosting Adaptation:** Handled challenges with cloud storage constraints by configuring local hosting infrastructure, compressing internal packages and environments to ensure smooth end-to-end execution.

---

## 📊 Model Performance & Evaluation

We trained and compared two models on the test set (`20%` split):

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 0.4504 | 1.1246 | 0.5490 (54.9%) |
| **Random Forest Regressor (Winner)** | 2.6298 | 7.5041 | **0.7876 (78.7%)** |

> **Conclusion & Justification:** The **Random Forest Regressor** was chosen as the winning model because it significantly outperformed the linear baseline across all evaluation metrics. Real estate prices depend heavily on non-linear interaction terms (e.g., combined effects of location, carpet area, and furnishing status), which tree-based ensemble models capture exceptionally well. (Bonus 5-Fold Cross-Validation confirmed model stability and generalizability without overfitting).

---

## 🛠️ Tech Stack

* **Machine Learning & Data:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
* **Backend API:** FastAPI, Uvicorn, Pydantic, Pytest, HTTPX
* **Frontend UI:** React, TypeScript, Vite, React Router
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
house-price-project/
│
├── notebooks/                   # Jupyter notebooks for EDA and training
│   ├── data/                    # Raw dataset location (house_prices.csv)
│   └── house_price_model.ipynb  # Full notebook pipeline
│
├── backend/                     # FastAPI backend service
│   ├── app/
│   │   ├── api/routes/          # Prediction and health check endpoints
│   │   ├── core/                # Configuration settings
│   │   ├── schemas/             # Pydantic validation models
│   │   ├── services/            # Preprocessing and model inference
│   │   └── utils/               # Logging configurations
│   ├── models/                  # Saved model (.pkl) and locations (.json)
│   ├── tests/                   # Pytest suite
│   ├── requirements.txt         # Pinned Python dependencies
│   └── Dockerfile               # Container configuration
│
├── frontend/                    # React + Vite frontend application
│   ├── src/
│   │   ├── api/                 # API fetch wrappers
│   │   ├── components/          # Input form components
│   │   ├── pages/               # Home and Result pages
│   │   └── types/               # TypeScript interfaces
│   └── package.json
│
└── README.md                    # Project Documentation




⚙️ Setup and Installation Instructions

Prerequisites
Python 3.11+
  Node.js 18+ & npm
 Git

1. Clone the Repository

git clone [https://github.com/Rokaia-Rezk/house-price-predictor.git](https://github.com/Rokaia-Rezk/house-price-predictor.git)
cd house-price-predictor

2. Backend Setup
Navigate to the backend directory, set up a virtual environment, install requirements, and run the server:

cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Run FastAPI server
uvicorn app.main:app --reload

The backend API will be accessible at http://localhost:8000 (Swagger docs at http://localhost:8000/docs)

3. Frontend Setup
Open a new terminal window, navigate to the frontend directory, install dependencies, and start the development server:

cd frontend
npm install
npm run dev

The frontend application will run at http://localhost:5173


🔌 API Reference
GET /health

Response: {"status": "ok"}

POST /predict

Request Body Example:

{
  "location": "Bhiwadi",
  "carpet_area_sqft": 1200.0,
  "floor_num": 2,
  "bathroom": 2,
  "balcony": 1,
  "furnishing": "Semi-Furnished",
  "transaction": "Resale",
  "ownership": "Freehold",
  "facing": "North"
}

Response Example:

{
  "predicted_price": 42842192.25
}
