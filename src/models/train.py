# src/models/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(df: pd.DataFrame, config: dict):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_params = config["model"]["xgboost"]

    model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=ratio,
        random_state=config["model"]["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
