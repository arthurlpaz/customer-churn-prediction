from __future__ import annotations

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

config_path = "src/config/config.yaml"


def _build_model(model_name: str, model_cfg: dict, random_state: int):
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install with `pip install xgboost` or use model.name=random_forest"
            ) from exc

        return XGBClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            min_child_weight=model_cfg["min_child_weight"],
            reg_lambda=model_cfg["reg_lambda"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    return RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
        class_weight=model_cfg.get("class_weight", None),
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(X, y, test_size=0.2, random_state=42):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"].get("name", "random_forest")
    model_cfg = config["model"][model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = _build_model(model_name, model_cfg, random_state=random_state)
    model.fit(X_train, y_train)

    print(f"Model selected: {model_name}")

    return model, X_test, y_test
