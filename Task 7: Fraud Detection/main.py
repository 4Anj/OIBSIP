import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import IsolationForest
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the dataset
df= pd.read_csv(r"C:\Users\anjan\Downloads\creditcard.csv")
print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())

df.drop(['Time'], axis=1, inplace=True)
# Scale 'Amount' column
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df.drop(['Amount'], axis=1, inplace=True)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Initialize models
log_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier(random_state=42)
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

for name, model in models.items():
    print(f"\n{name}")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

# Confusion Matrix (Random Forest)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, forest_model.predict(X_test)), 
            annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xticks(ticks=[0.5, 1.5], labels=['Normal', 'Fraud'])
plt.yticks(ticks=[0.5, 1.5], labels=['Normal', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(forest_model, X_test, y_test)
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Feature Importance
importances = forest_model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Anomaly Detection using Isolation Forest
iso_model = IsolationForest(contamination=0.01, random_state=42)
iso_model.fit(X_train)
scores = iso_model.decision_function(X_test)

joblib.dump(model, 'fraud_model.pkl')
print("Model saved as fraud_model.pkl")

# Load trained model
model = joblib.load("fraud_model.pkl")

# Define API app
app = FastAPI()

# Input schema
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
        transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
        transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
        transaction.V26, transaction.V27, transaction.V28, transaction.Amount
    ]])
    
    prediction = model.predict(data)[0]
    result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
    return {"prediction": int(prediction), "result": result}
