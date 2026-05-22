from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "churn_pipeline.pkl"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "ContractType",
    "InternetService",
    "TechSupport",
]


class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: Literal["Male", "Female"]
    tenure: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    contract_type: Literal["Month-to-Month", "One-Year", "Two-Year"]
    internet_service: Literal["DSL", "Fiber Optic", "None", "Unknown"]
    tech_support: Literal["Yes", "No"]


app = FastAPI(
    title="Customer Churn Prediction API",
    description="A simple FastAPI demo for predicting customer churn.",
    version="1.0.0",
)


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Run `python train_model.py` first.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    model = load_model()
    input_data = pd.DataFrame(
        [
            {
                "Age": customer.age,
                "Gender": customer.gender,
                "Tenure": customer.tenure,
                "MonthlyCharges": customer.monthly_charges,
                "ContractType": customer.contract_type,
                "InternetService": customer.internet_service,
                "TechSupport": customer.tech_support,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(input_data)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": probability,
    }
