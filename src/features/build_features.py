# src/features/build_features.py
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df
