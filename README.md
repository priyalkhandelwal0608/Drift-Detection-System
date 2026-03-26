
# Drift Detection System

An MLOps pipeline designed to monitor machine learning models in production, detect data/concept drift, and trigger retraining to maintain model accuracy.

##  Project Structure

| Directory | Description |
| :--- | :--- |
| **api/** | Contains `app.py` for serving the model via a REST API. |
| **data/** | Scripts for generating synthetic training/production data and stored `.csv` files. |
| **model/** | Stores the trained `model.pkl` and the logic in `train_model.py`. |
| **monitoring/** | The core engine (`drift_detection.py`) that compares production vs. reference data. |
| **retraining/** | Automated scripts to refresh the model when drift is detected. |

---
##  Key Components

### **Drift Detection**
The `monitoring/drift_detection.py` script identifies shifts between `reference_data.csv` (the data the model was trained on) and incoming `production_data.csv`. This ensures the model isn't making "blind" predictions on data it no longer understands.

### **Retraining Loop**
When the monitor flags significant drift (e.g., using a Kolmogorov-Smirnov test), `retraining/retrain.py` is triggered to update `model.pkl` using the most recent production data samples.

---
##Installation 
# Start monitoring for data drift
python monitoring/monitor.py

# Manually trigger retraining if drift is detected
python retraining/retrain.py
-pip install -r requirements.txt
-python data/generate_data.py
-python data/generate_production_data.py
-python model/train_model.py
-python monitoring/monitor.py
-python retraining/retrain.py
-uvicorn api.app:app --reload
