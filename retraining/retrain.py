import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
import mlflow.sklearn

REFERENCE = os.path.join("data", "reference_data.csv")
MODEL_PATH = os.path.join("model", "model.pkl")

if not os.path.exists(REFERENCE):
    raise FileNotFoundError(f"Reference dataset not found at {REFERENCE}")

ref = pd.read_csv(REFERENCE)
X = ref[["transaction_amount", "account_age_days", "num_transactions"]]
y = ref["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    
    mlflow.log_param("retrain", True)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model retrained and saved at {MODEL_PATH}")
    print(f"Accuracy: {acc}")
