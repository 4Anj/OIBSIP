# Wine Quality Classification with Machine Learning
This project performs multiclass classification on wine samples using chemical features to predict wine quality scores. Multiple models are evaluated: Random Forest, Support Vector Classifier (SVC), and Stochastic Gradient Descent (SGD). The analysis includes data exploration, normalization, training, evaluation, and performance comparison through visualizations.

### Dataset
Filename: WineQT.csv   https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

Contains physicochemical test results (inputs) and wine quality scores (target)

Target Variable: quality (score from 3 to 8)

### Tools & Libraries Used
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```
Install all dependencies via:
```bash
pip install -r requirements.txt
```

### Workflow Overview
1. Data Exploration & Preprocessing
Explored dataset shape, data types, and distribution of target labels

Visualized: Wine quality distribution
Correlation heatmap among features
Scaled features using StandardScaler to standardize input for ML algorithms

2. Model Building & Training
Random Forest Classifier
Ensemble-based model
Handles class imbalance and noisy features well

Support Vector Classifier (SVC)
Effective in high-dimensional space
Works well for moderately-sized datasets

Stochastic Gradient Descent (SGD)
Efficient with large-scale and sparse datasets
Online learning via gradient descent

3. Model Evaluation
Metrics Used:
Classification Report (precision, recall, F1-score)
Confusion Matrix for each model

Visualization:
Heatmaps for each model’s confusion matrix to assess prediction performance

```python
sns.heatmap(confusion_matrix(y_test, model_preds), annot=True)
```

### Output 
| Model         | Accuracy | Weighted F1-Score | Observations                                |
| ------------- | -------- | ----------------- | ------------------------------------------- |
| Random Forest | **0.69** | **0.68**          | Best performing model overall               |
| SVC           | 0.66     | 0.64              | Decent, but struggles with minority classes |
| SGD           | 0.61     | 0.59              | Worst overall performance                   |


Random Forest outperforms others in most metrics, making it a strong baseline.

🔍 Key Insights
Wine quality scores are imbalanced, with most samples clustered between 5 and 6.

Random Forest provides better generalization and handles multiple classes effectively.

Features like alcohol, volatile acidity, and sulphates show stronger correlations with quality.
