# Retail Sales Analysis using Python
This project performs a comprehensive analysis of a retail sales dataset using Python. The workflow includes data cleaning, exploratory data analysis (EDA), time series analysis, customer and product insights, and visualizations.

## Dataset
The dataset used is an Excel file:
link: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset
retail_sales_dataset.xlsx

the dataset has the following key columns:

Date   Customer ID   Gender   Product Category   Total Amount

## Tools & Libraries Used
Pandas – for data manipulation and cleaning

NumPy – for numerical operations

Matplotlib – for basic visualizations

Seaborn – for statistical plots

## Features and Analysis
1. Data Cleaning
Checked data shape, info, and missing values.

Converted the Date column to datetime format.

Removed unwanted whitespaces from column names.

2. Descriptive Statistics
Calculated Mean, Median, Mode, and Standard Deviation for numeric columns.

3. Time Series Analysis
Extracted Month and Year from Date.

Grouped total sales by Year and Month.

Visualized monthly sales trends using a line plot.

4. Customer & Product Analysis
Identified Top 10 customers based on total sales.

Found Top 10 product categories by total sales.

Plotted customer gender distribution.

5. Visualizations
Bar Chart for sales by product category.

Line Chart for sales over time.

Heatmap to explore correlations among numeric features.

## How to Run
1. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```
2. Run the Python script (sales_analysis.py)
3. Ensure the dataset path is correct
