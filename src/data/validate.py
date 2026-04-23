import pandas as pd


def validate_data(df: pd.DataFrame):
    assert df is not None, "Dataset vazio"
    assert "Churn" in df.columns, "Target não encontrado"
