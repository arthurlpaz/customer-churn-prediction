from sklearn.model_selection import train_test_split
from src.models.pipeline import build_pipeline


def train_model(df, config):
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]

    pipeline = build_pipeline(df, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test
