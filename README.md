# 🏡 Real Estate Price Predictor (End-to-End ML Web Application)

A robust, production-ready, full-stack Machine Learning web application designed to predict real estate prices. This project covers the complete machine learning lifecycle: from raw data ingestion, meticulous Exploratory Data Analysis (EDA), advanced data cleaning, feature engineering, and model pipeline training, to serving via a scalable **FastAPI** backend (deployed on **Railway**) and interacting through a responsive, custom-styled **React + TypeScript + Vite** frontend (deployed on **Vercel**).

🔗 **Live Application:** [Open Real Estate Predictor](https://house-price-predictor-qn6p16vrv-rokaia-rezk1.vercel.app/)  
🔌 **Live Backend API (Swagger UI):** [Explore API Docs](https://alluring-wisdom-production-6322.up.railway.app/docs)
📊 **Dataset Source:** [Kaggle Real Estate Dataset (by Juhi Bhojani)](https://www.kaggle.com/datasets/juhibhojani/house-price) *(~187,000 real estate listings)*

---

## 🚀 Project Overview & Engineering Journey

Building this project involved tackling messy real-world data and building a robust production pipeline from scratch:
* **Data Ingestion & Inspection:** Loaded and analyzed a massive dataset of ~187,000 real estate listings (`house_prices.csv`) sourced from Kaggle.
* **Exploratory Data Analysis (EDA):** Generated visual insights including target price distributions (using Log-Scale), scatter plots of price vs. carpet area, bar charts for top locations, and box plots for furnishing status.
* **Data Cleaning & Feature Engineering:**
  * Parsed complex text strings in financial amounts (converting Lacs and Crores to exact numerical values) and carpet areas (normalizing sqft and sqm).
  * Handled complex floor string layouts ("3 out of 10", "Ground", "Basement").
  * Grouped high-cardinality categorical features (retaining top locations and mapping the rest to "other" to avoid the curse of dimensionality).
  * Dropped redundant columns and removed extreme outliers based on price-per-sqft (1st and 99th percentiles).
* **Modeling & Pipeline Integration:** Trained and optimized robust regression models using Scikit-Learn pipelines, wrapping preprocessors and estimators seamlessly.

---

## 📊 Model Performance & Evaluation

We evaluated multiple models on a 20% test split to ensure maximum prediction accuracy:

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 0.4504 | 1.1246 | 0.5490 (54.9%) |
| **Random Forest Regressor (Winner)** | 2.6298 | 7.5041 | **0.7876 (78.7%)** |

* **Justification:** The Random Forest Regressor significantly outperformed the linear baseline. Real estate valuations depend heavily on non-linear interactions (such as the combined effect of location tier, exact carpet area, and furnishing status), which tree-based ensemble models capture exceptionally well. 5-fold cross-validation confirmed robust generalization without overfitting.

---

## 🛠️ Tech Stack & Architecture

* **Machine Learning & Data Processing:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
* **Backend API:** FastAPI, Uvicorn, Pydantic, CORS Middleware, JSON-based dynamic location routing
* **Frontend UI:** React, TypeScript, Vite, Responsive Mobile-First CSS Design (Two-Column Adaptive Grid)
* **Deployment & Cloud Infrastructure:** 
  * **Backend:** Railway (Dockerized / Python runtime container)
  * **Frontend:** Vercel (Global Edge Network)
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
house-price-project/
│
├── notebooks/                   # Jupyter notebooks for EDA and training
│   ├── data/                    # Raw dataset location (house_prices.csv)
│   └── house_price_model.ipynb  # Full data pipeline & model experimentation
│
├── backend/                     # FastAPI backend service
│   ├── app/
│   │   ├── api/routes/          # Prediction and health check endpoints
│   │   ├── core/                # Configuration settings
│   │   ├── schemas/             # Pydantic validation models
│   │   ├── services/            # Preprocessing and model inference
│   │   └── utils/               # Logging configurations
│   ├── models/                  # Serialized model (.pkl) and locations (.json)
│   ├── tests/                   # Pytest suite
│   ├── requirements.txt         # Pinned Python dependencies
│   └── Dockerfile               # Container configuration
│
├── frontend/                    # React + TypeScript + Vite application
│   ├── src/
│   │   ├── api/                 # API fetch wrappers & dynamic endpoint resolution
│   │   ├── components/          # Responsive UI components & mobile grid
│   │   ├── pages/               # Main predictor interface
│   │   └── types/               # TypeScript interfaces
│   └── package.json
│
└── README.md                    # Project Documentation



📱 Frontend & Mobile Responsiveness Optimization
The user interface was meticulously engineered to provide an app-like experience across all device sizes:

Adaptive Grid Architecture: Implemented a refined two-column layout on mobile viewports, maintaining compact padding, clean margins, and readable typography to mirror desktop experiences without horizontal overflow.

Custom Styling: Enhanced with an elegant pastel-purple aesthetic, animated floral background elements, custom input focus states, and a dedicated branded header (✨ ENG. Rokaia Rezk).


🔌 API Reference & Endpoints
1. Health Check
GET /health

Response Example:

{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "GradientBoosting_Regressor_Pipeline"
}



2. Predict Price
POST /predict

Request Body Example:

{
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


Response Example:

{
  "prediction": 6400000.0
}


⚙️ Local Setup and Installation
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
uvicorn app.main:app --reload


The API will be live at http://localhost:8000 (Interactive Swagger Docs at http://localhost:8000/docs).



3. Frontend Setup
Open a new terminal window, navigate to the frontend directory, install dependencies, and start the development server:

cd frontend
npm install
npm run dev

The frontend application will run at http://localhost:5173.



## 💡 Author
* **Rokaia Hassan Mokhtar Rezk**  
* **AI Solutions & Data Analytics Engineer** (Specializing in Machine Learning, Prompt Engineering, Human Behavior Analysis & Psychology)
