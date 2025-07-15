# Oasis Infobytes Data Analytics Internship – Final Report
Data Analytics Internship at Oasis Infobytes.
This internship provided the opportunity to work on real-world datasets, apply machine learning techniques, clean data, build predictive models, and perform insightful data analysis.

### Internship Tasks Overview
| Task No. |    Project Title                                   |    Topics Covered                                          |
| -------- | -------------------------------------------------- | ---------------------------------------------------------- |
| 1        | **EDA on Retail Sales Data**                       | Data Cleaning, EDA, Visualization                          |
| 2        | **Customer Segmentation**                          | Clustering, Feature Scaling, KMeans                        |
| 3        | **Sentiment Analysis**                             | NLP, TF-IDF, Naive Bayes, WordCloud                        |
| 4        | **Cleaning Data**                                  | Data Preprocessing, Handling Missing & Categorical Data    |
| 5        | **Predicting House Prices with Linear Regression** | Regression, Feature Engineering, Model Evaluation          |
| 6        | **Wine Quality Prediction**                        | Classification, Random Forest, SVM, SGD                    |
| 7        | **Fraud Detection**                                | Imbalanced Data, Random Forest, Logistic, API Deployment   |
| 8        | **Unveiling the Android App Market**               | Exploratory Analysis, Revenue Estimation, TextBlob NLP     |
| 9        | **Autocomplete and Autocorrect Data Analytics**    | NLP, Norvig Spell Correction, Bigrams, Accuracy Comparison |


### Project Highlights
1. EDA on Retail Sales Data
Performed exploratory data analysis on retail sales.

Analyzed revenue, popular products, regions, and sales trends.

Visualized insights using bar plots and pie charts.

2. Customer Segmentation
Applied KMeans clustering on customer attributes.

Scaled features for balanced distance computation.

Visualized clusters to identify key customer groups.

3. Sentiment Analysis
Cleaned Twitter data and applied TF-IDF vectorization.

Built a Naive Bayes classifier to predict sentiment.

Visualized class distribution, confusion matrix, and word clouds.

4. Cleaning Data
Worked on converting datatypes, handling nulls, cleaning symbols.

Applied regex and basic imputation.

Ensured high-quality structured datasets for analysis.

5. Predicting House Prices with Linear Regression
One-hot encoded categorical features.

Trained and evaluated a linear regression model.

Analyzed residuals and visualized feature importance.

6. Wine Quality Prediction
Compared RandomForest, SVM, and SGD classifiers.

Evaluated performance using classification metrics.

Visualized confusion matrices and class distribution.

7. Fraud Detection
Used Logistic Regression, Decision Tree, and RandomForest.

Applied Isolation Forest for anomaly detection.

Deployed fraud prediction model using FastAPI.

8. Unveiling the Android App Market
Cleaned Play Store data and estimated revenue.

Visualized top-rated, most installed, and paid apps.

Performed sentiment analysis on user reviews using TextBlob.

9. Autocomplete and Autocorrect Data Analytics
Built Norvig-based autocorrect algorithm.

Developed bigram-based autocomplete system.

Compared performance with TextBlob correction.

### Accuracy Snapshot
| Project                          | Accuracy / Score |
| -------------------------------- | ---------------- |
| Sentiment Classification (NB)    | \~70%            |
| House Price Prediction (Linear)  | Good RMSE, R²    |
| Wine Quality Classification (RF) | \~69% Accuracy   |
| Fraud Detection (RF AUC)         | \~0.98 AUC       |
| Autocorrect (Norvig)             | 57.38%           |
| Autocomplete                     | 89.16%           |


### Tech Stack Used
Languages: Python

ML Libraries: Scikit-learn, NLTK, TextBlob

Data Wrangling: Pandas, NumPy, Regex

Visualization: Matplotlib, Seaborn, Plotly

Deployment: FastAPI, Joblib

Other Tools: WordCloud, TF-IDF, StandardScaler, KMeans

### How to Run
1. Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob nltk wordcloud fastapi uvicorn joblib plotly
```
2. Download the datasets and update local file paths.

3. Run each task’s Python file individually.

For API (fraud detection), run:
```bash
uvicorn your_api_file:app --reload
```

### Learning Outcomes
Understood the end-to-end data science pipeline.

Applied machine learning to real-world problems.

Explored text analytics and NLP from scratch.

Gained practical experience in data cleaning, preprocessing, model building, and API deployment.

Developed confidence in using industry tools like Scikit-learn, FastAPI, TextBlob, and Plotly.

### Acknowledgement
This internship with Oasis Infobytes has been an excellent opportunity to learn, apply, and grow as a Data Analyst and budding Data Scientist.
