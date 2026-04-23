from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml

config_path = "src/config/config.yaml"


def train_model(X, y, test_size=0.2, random_state=42):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["random_forest"]["n_estimators"],
        max_depth=config["model"]["random_forest"]["max_depth"],
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
