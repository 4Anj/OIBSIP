# Credit Card Fraud Detection with Machine Learning & FastAPI
This project builds a fraud detection system using machine learning models trained on real-world transaction data. It includes exploratory analysis, model training, evaluation, anomaly detection, and a live REST API using FastAPI to predict if a transaction is fraudulent or legitimate.

### Dataset
Source: Kaggle - Credit Card Fraud Detection    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Size: 284,807 transactions with 31 features
Target Column: Class → 0 = Legitimate, 1 = Fraudulent
Highly imbalanced dataset (fraud: 492, normal: 284,315)

### Tools & Libraries Used
pandas, numpy, matplotlib, seaborn
scikit-learn (models, metrics, preprocessing)
joblib for model serialization
FastAPI + pydantic for the REST API
IsolationForest for unsupervised anomaly detection

### Workflow Summary
1. Data Preprocessing
Dropped Time column as irrelevant.
Scaled Amount using StandardScaler and renamed it to Amount_scaled.
Handled class imbalance by using stratified split.

2. Models Trained
Model	Description
Logistic Regression	Baseline linear model
Decision Tree	Simple, interpretable tree
Random Forest	Robust, ensemble-based tree model

```python
models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}
```

3. Model Evaluation
Each model is evaluated using:
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix
ROC AUC Score
ROC Curve Visualization
Feature Importance (for Random Forest)

### Metrics
Model	AUC-ROC Score	Best For
Logistic Regression	~0.96	Interpretability
Decision Tree	~0.88	Simplicity
Random Forest	~0.99	Best Performance

### Anomaly Detection
Used Isolation Forest as an unsupervised technique to detect anomalies beyond supervised learning.

```python
iso_model = IsolationForest(contamination=0.01)
```

### FastAPI REST API
How to Run the API
Install FastAPI and Uvicorn:
```bash
pip install fastapi uvicorn joblib scikit-learn
```

Run the server:
```bash
uvicorn fraud_detection:app --reload
```

POST Endpoint:
```nginx
POST http://127.0.0.1:8000/predict
```

### JSON Input Format
```json
{
  "V1": -1.359,
  "V2": -0.072,
  ...
  "V28": 0.021,
  "Amount": 149.62
}
```

### Response Format
```json
{
  "prediction": 0,
  "result": "Legitimate Transaction"
}
```

### Recommendations
Use SMOTE or undersampling for further balancing
Experiment with XGBoost, LightGBM for advanced modeling
Integrate authentication for production API
Containerize with Docker for deployment

### Requirements
requirements.txt:
```nginx

pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
fastapi
uvicorn
```
