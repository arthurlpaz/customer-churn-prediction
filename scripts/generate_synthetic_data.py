import argparse

import numpy as np
import pandas as pd


def build_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], rows),
            "SeniorCitizen": rng.integers(0, 2, rows),
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows),
            "tenure": rng.integers(0, 72, rows),
            "PhoneService": rng.choice(["Yes", "No"], rows),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], rows),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], rows),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], rows),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], rows),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], rows),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], rows),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], rows),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], rows),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], rows),
            "PaperlessBilling": rng.choice(["Yes", "No"], rows),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                rows,
            ),
            "MonthlyCharges": rng.uniform(18.0, 120.0, rows).round(2),
            "TotalCharges": rng.uniform(18.0, 9000.0, rows).round(2),
            "Churn": rng.choice(["Yes", "No"], rows, p=[0.27, 0.73]),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telco churn dataset")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows")
    args = parser.parse_args()

    dataset = build_dataset(rows=args.rows)
    dataset.to_csv(args.output, index=False)
    print(f"Synthetic dataset written to {args.output} with shape={dataset.shape}")
