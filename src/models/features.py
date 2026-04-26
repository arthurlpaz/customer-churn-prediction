import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df
