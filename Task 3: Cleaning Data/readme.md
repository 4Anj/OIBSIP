# Airbnb NYC 2019 Data Cleaning & Outlier Detection
This project focuses on cleaning, preprocessing, and visualizing outliers in the Airbnb NYC 2019 dataset. The analysis identifies and removes anomalies in pricing and stay durations to improve data quality for future insights or machine learning tasks.

### Dataset
Filename: AB_NYC_2019.csv   https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Contains listings information for Airbnb stays in New York City (2019).

Key columns used:
price: Price per night (target for outlier analysis)

minimum_nights: Minimum nights to book

neighbourhood_group, room_type, last_review, etc.

### Libraries Used
```bash
pandas
matplotlib
seaborn
```

### Steps Performed
1. Data Loading & Initial Inspection
Loaded CSV file and printed the first 10 records
Inspected missing values and summary statistics

2. Data Cleaning
Dropped null values and duplicates

Normalized column names (lowercase, underscores)

Converted:

last_review to datetime, price to float, Standardized strings in object columns

3. Outlier Detection and Removal
Before Cleaning:
Plotted boxplots of price and minimum_nights
Observed extreme values and long tails

### IQR Method:
Calculated Interquartile Range (IQR) for price
Removed records outside 1.5 * IQR and rows with minimum_nights > 365

```python
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
```

### After Cleaning:
Replotted boxplots for price and minimum_nights
Significant reduction in skewness and noise

### Visualizations
Boxplot – Price Distribution (Before & After Cleaning)

Boxplot – Minimum Nights Distribution (Before & After Cleaning)

These plots helped visually validate the effectiveness of outlier removal.

### Key Insights
NYC Airbnb listings had significant price outliers likely due to rare or luxury accommodations.

A small number of listings had extremely long minimum stays (e.g., 999+ nights).

Removing these outliers resulted in a more usable, balanced dataset.
