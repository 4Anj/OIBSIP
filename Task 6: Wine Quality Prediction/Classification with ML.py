import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\anjan\Downloads\WineQT.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df['quality'].value_counts())

sns.countplot(x='quality', data=df, palette='viridis')
plt.title("Wine Quality Distribution")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Prepare the data
X = df.drop('quality', axis=1)     # features
y = df['quality']                  # target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))

# Support Vector Classifier and Stochastic Gradient Descent Classifier
svc = SVC()
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
print("SVC Classification Report:")
print(classification_report(y_test, svc_preds))

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd_preds = sgd.predict(X_test)
print("SGD Classification Report:")
print(classification_report(y_test, sgd_preds))

# Visualizing the confusion matrices for each model
models = {'Random Forest': rf_preds, 'SGD': sgd_preds, 'SVC': svc_preds}
for name, preds in models.items():
    print(f"\n{name} Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
