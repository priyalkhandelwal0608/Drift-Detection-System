# ML Experiment Tracking with MLflow

An **MLOps pipeline** to monitor machine learning models in production, detect data/concept drift, trigger retraining, and track experiments with **MLflow** to maintain model performance and reproducibility.

---

##  Project Overview

This project demonstrates a **full ML lifecycle**:

1. **Data Generation** – Create synthetic reference and production datasets.  
2. **Model Training** – Train a Random Forest model on reference data.  
3. **Drift Detection** – Monitor incoming production data for shifts compared to reference data.  
4. **Retraining** – Automatically retrain the model when drift is detected.  
5. **Experiment Tracking** – Log parameters, metrics, and models with MLflow for easy comparison.  
6. **Model Serving** – Provide predictions through a REST API using FastAPI.

---

##  Project Structure

| Directory | Description |
| :--- | :--- |
| **api/** | REST API serving the model (`app.py`) with a frontend template. |
| **data/** | Scripts for generating synthetic datasets (`generate_data.py`, `generate_production_data.py`). |
| **model/** | Model training logic (`train_model.py`) and saved model (`model.pkl`). |
| **monitoring/** | Drift detection engine (`drift_detection.py`, `monitor.py`). |
| **retraining/** | Automated retraining script (`retrain.py`). |
| **templates/** | HTML templates for API frontend. |
| **static/** | CSS/JS files for UI styling. |
| **mlruns/** | MLflow tracking folder (auto-generated). |

---

##  Key Components

### 1. Drift Detection

- **File:** `monitoring/drift_detection.py`  
- Compares **reference data** (training) and **production data**.  
- Uses the **Kolmogorov-Smirnov test** to detect shifts in feature distributions.  
- If drift is detected, triggers the retraining pipeline.

### 2. Retraining Loop

- **File:** `retraining/retrain.py`  
- Retrains the model on the latest data.  
- Saves updated `model.pkl`.  
- Logs experiment parameters, metrics, and model artifacts to **MLflow**.

### 3. Experiment Tracking with MLflow

- **Track every run:** `train_model.py` and `retrain.py` log metrics (`accuracy`), parameters (`model_type`, retrain status), and artifacts (saved model).  
- **Compare runs:** Use MLflow UI to analyze improvements, retraining impact, or hyperparameter changes.  
- **Model registry:** Optionally, register best-performing models for production deployment.

### 4. Model Serving

- **File:** `api/app.py`  
- **Framework:** FastAPI  
- Provides a **web form** to input features:  
  - `transaction_amount`  
  - `account_age_days`  
  - `num_transactions`  
- Returns **Fraudulent** or **Legitimate Transaction** predictions.  
- Optional: API can log predictions to MLflow for monitoring.

---

##  Installation & run
 - pip install -r requirements.txt
 - python data/generate_data.py
 - python data/generate_production_data.py
 - python model/train_model.py
 - python monitoring/monitor.py
 - python retraining/retrain.py
 - uvicorn api.app:app --reload
 - mlflow ui 
