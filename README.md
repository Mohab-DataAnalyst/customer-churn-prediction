# Customer Churn Prediction

This project predicts whether a customer is likely to churn based on customer profile, contract, service, tenure, and billing information. It includes exploratory analysis, model training, and simple deployment demos using FastAPI and Streamlit.

## Project Overview

Customer churn is a common business problem where companies want to identify customers who may leave so they can take proactive retention actions. In this project, I built a machine learning pipeline that preprocesses customer data and predicts churn as `Yes` or `No`.

## Dataset

The dataset contains customer-level information such as:

- Age
- Gender
- Tenure
- Monthly charges
- Contract type
- Internet service
- Total charges
- Tech support
- Churn status

## Workflow

1. Loaded and inspected the customer churn dataset.
2. Checked duplicates and missing values.
3. Filled missing `InternetService` values with `Unknown`.
4. Explored churn patterns by customer and service features.
5. Built preprocessing pipelines for numerical and categorical columns.
6. Trained classification models and selected the best model using F1 score.
7. Saved the trained pipeline as `churn_pipeline.pkl`.
8. Added FastAPI and Streamlit demo apps for deployment.

## Model Features

The deployed model uses these inputs:

- `Age`
- `Gender`
- `Tenure`
- `MonthlyCharges`
- `ContractType`
- `InternetService`
- `TechSupport`

## Tech Stack

- Python
- pandas
- scikit-learn
- joblib
- FastAPI
- Streamlit
- Jupyter Notebook

## Project Structure

```text
.
+-- Churn Prediction Project.ipynb
+-- customer_churn.csv
+-- train_model.py
+-- churn_pipeline.pkl
+-- app.py
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

Run the FastAPI app:

```bash
uvicorn app:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## API Example

Send a `POST` request to `/predict`:

```json
{
  "age": 35,
  "gender": "Female",
  "tenure": 12,
  "monthly_charges": 72.5,
  "contract_type": "Month-to-Month",
  "internet_service": "Fiber Optic",
  "tech_support": "No"
}
```

Example response:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.76
}
```

## What I Learned

This project helped me practice the end-to-end data science workflow: data cleaning, exploratory data analysis, preprocessing, model training, model evaluation, and building simple deployment interfaces for demos.
