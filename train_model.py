from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "customer_churn.csv"
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
TARGET_COLUMN = "Churn"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["InternetService"] = df["InternetService"].fillna("Unknown")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    numerical_features = ["Age", "Tenure", "MonthlyCharges"]
    categorical_features = ["Gender", "ContractType", "InternetService", "TechSupport"]

    preprocessor = ColumnTransformer(
        [
            ("numerical", StandardScaler(), numerical_features),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"F1 score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
