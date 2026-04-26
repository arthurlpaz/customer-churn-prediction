# 🚀 Customer Churn Prediction — Production ML Pipeline

Complete Machine Learning project for **customer churn prediction**, developed with a focus on **production-grade engineering best practices**.

---

# 📌 Goal

Predict the probability of customer churn using:

- Structured ML pipeline
- Optimized XGBoost
- Integrated feature engineering
- Real-time inference API
- Experiment tracking with MLflow
- Automated CI/CD

---

# 🧱 Project Architecture

```bash
src/
│
├── api/                  # FastAPI application
│   ├── app.py
│   └── schema.py
│
├── data/                 # Data ingestion
│   └── load_data.py
│
├── models/
│   ├── pipeline.py       # Production-ready pipeline
│   ├── train.py
│   └── features.py       # Feature engineering
│
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── config/
│   └── config.yaml
│
artifacts/
│   └── models/           # Saved model artifacts
│
.github/
│   └── workflows/
│       └── ml_pipeline.yml  # CI/CD
```

---

# ⚙️ Technologies

- Python 3.10+
- Scikit-learn
- XGBoost
- FastAPI
- MLflow
- Docker
- GitHub Actions (CI/CD)

---

# 🧠 Machine Learning Pipeline

End-to-end pipeline:

```
Input → Feature Engineering → Preprocessing → Model → Prediction
```

### ✔️ Highlights

- Feature engineering inside the pipeline
- Automatic data handling
- OneHotEncoding + Scaling
- Class imbalance handled via `scale_pos_weight`
- Threshold tuning

---

# 🔧 Configuration (`config.yaml`)

```yaml
model:
  test_size: 0.2
  random_state: 42
  threshold: 0.3

  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    eval_metric: logloss
```

---

# 🚀 How to run the project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train the model

```bash
python main.py
```

---

## 3. Run MLflow UI

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## 4. Start the API

```bash
uvicorn src.api.app:app --reload
```

Open Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

# 🧪 Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1020.6
}
```

---

# 📊 MLflow

This project uses MLflow for:

- Experiment tracking
- Logging metrics (ROC-AUC, Recall, F1)
- Model versioning

Generated files:

```
mlflow.db
mlruns/
```

---

# 🐳 Docker

Build:

```bash
docker build -t churn-api .
```

Run:

```bash
docker run -p 8000:8000 churn-api
```

---

# 👨‍💻 Author

Arthur da Paz

---

