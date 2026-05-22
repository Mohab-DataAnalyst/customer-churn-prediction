from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate whether the customer is likely to churn.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0)
    contract_type = st.selectbox(
        "Contract type",
        ["Month-to-Month", "One-Year", "Two-Year"],
    )
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber Optic", "None", "Unknown"],
    )
    tech_support = st.selectbox("Tech support", ["Yes", "No"])
    submitted = st.form_submit_button("Predict churn")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "ContractType": contract_type,
                "InternetService": internet_service,
                "TechSupport": tech_support,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(input_data)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is likely to stay")

    if probability is not None:
        st.metric("Churn probability", f"{probability:.1%}")
