import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
import mlflow.sklearn

# Paths
DATA_PATH = os.path.join("data", "reference_data.csv")
MODEL_PATH = os.path.join("model", "model.pkl")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Reference dataset not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
X = data[["transaction_amount", "account_age_days", "num_transactions"]]
y = data["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    # Save locally
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved at {MODEL_PATH}")
    print(f"Accuracy: {acc}")
