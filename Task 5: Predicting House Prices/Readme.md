# Housing Price Prediction with Linear Regression
This project builds a Linear Regression model to predict housing prices using various features such as location attributes, furnishing status, and amenities. The pipeline involves data preprocessing, feature encoding, scaling, model evaluation, and multiple visualizations to interpret results and residuals.

### Dataset
Filename: Housing.csv   https://www.kaggle.com/code/ashydv/housing-price-prediction-linear-regression

Feature Types:

Numerical: area, bedrooms, bathrooms, etc.

Categorical: mainroad, guestroom, basement, airconditioning, etc.

### Tools & Libraries
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Workflow Summary
1. Data Exploration
Displayed dataset structure, dimensions, and summary statistics

Checked for null values (none found)

Verified data types and structure

2. Data Preprocessing
One-Hot Encoding
Encoded categorical variables using pd.get_dummies() with drop_first=True to avoid multicollinearity

Feature Scaling
Standardized all features using StandardScaler for improved model convergence and interpretability

3. Model Building
Split the dataset into training (80%) and testing (20%)
Trained a Linear Regression model
Evaluated performance using:
Root Mean Squared Error (RMSE)
Mean Squared Error (MSE)
R² Score

### Visualizations
Actual vs Predicted Prices (Scatter Plot)
Visual check of how well predicted values align with actual test data
Ideal case: points close to the diagonal

#### Residuals vs Predicted
Helps assess bias in the model
A random scatter around 0 suggests good model fit

#### Distribution of Residuals
Visualizes how residuals are distributed (should resemble a normal distribution if model assumptions hold)

#### Feature Importance (Bar Plot)
Displays the magnitude of model coefficients for each feature
Indicates which features most strongly influence price

#### Correlation Heatmap
Shows correlation between all encoded features and price
Useful for multicollinearity detection and feature selection

### Model Evaluation Metrics
Metric	Value
RMSE	e.g., 114504.5
MSE	e.g., 13112772951.4
R²	e.g., 0.68 (68% variance explained)

### Key Insights
Area, number of bedrooms, and furnishing status have strong positive impact on house price.

Features like air conditioning and main road access also significantly increase value.

Residuals appear to be reasonably normally distributed with no major skew, suggesting a decent fit.
