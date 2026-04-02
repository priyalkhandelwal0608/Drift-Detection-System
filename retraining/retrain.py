import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
import mlflow.sklearn

# Paths
REFERENCE_PATH = os.path.join("data", "reference_data.csv")
PRODUCTION_PATH = os.path.join("data", "production_data.csv")
MODEL_PATH = os.path.join("model", "model.pkl")

# Load reference data
ref = pd.read_csv(REFERENCE_PATH)

# Load production data
if not os.path.exists(PRODUCTION_PATH):
    raise FileNotFoundError(f"Production dataset not found at {PRODUCTION_PATH}")

prod = pd.read_csv(PRODUCTION_PATH)

# Map predictions back to fraud column (optional: only if you have true labels)
# Here we assume production data doesn't have 'fraud' labels; if you have them, use them instead
if "fraud" not in prod.columns:
    prod["fraud"] = prod.get("prediction", 0)  # use model predictions as pseudo-labels

# Append production data to reference data
updated_ref = pd.concat([ref, prod[["transaction_amount", "account_age_days", "num_transactions", "fraud"]]], ignore_index=True)

# Save updated reference dataset
updated_ref.to_csv(REFERENCE_PATH, index=False)
print(f"Reference dataset updated with production data. Total rows: {len(updated_ref)}")

# Split features and target
X = updated_ref[["transaction_amount", "account_age_days", "num_transactions"]]
y = updated_ref["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retrain model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    acc = model.score(X_test, y_test)

    # Log retraining info
    mlflow.log_param("retrain", True)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save retrained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model retrained and saved at {MODEL_PATH}")
<<<<<<< HEAD
    print(f"Accuracy: {acc}")
=======
    print(f"Accuracy: {acc}")
>>>>>>> 0e5634b3d0a42a27607c4a0b743ec264719da898
