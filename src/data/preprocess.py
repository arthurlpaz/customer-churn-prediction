# src/data/preprocess.py
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df