import mlflow


def log_experiment(model, roc_auc):
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "model")
