# src/models/pipeline.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.models.features import add_features


def build_pipeline(df: pd.DataFrame, config: dict):
    df = df.copy()

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    if "Churn" in numeric_features:
        numeric_features.remove("Churn")
    if "Churn" in categorical_features:
        categorical_features.remove("Churn")

    if "customerID" in numeric_features:
        numeric_features.remove("customerID")
    if "customerID" in categorical_features:
        categorical_features.remove("customerID")

    ratio = (df["Churn"] == 0).sum() / (df["Churn"] == 1).sum()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    xgb_params = config["model"]["xgboost"]

    model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=ratio,
        random_state=config["model"]["random_state"],
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            ("feature_engineering", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
