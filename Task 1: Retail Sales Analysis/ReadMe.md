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

## Visualization Insights
1. Monthly Sales Trend (Time Series Line Plot)
Insight: The line plot of monthly sales over time reveals clear seasonal patterns, sales peaks, or possible growth/decline trends.

Specific months where sales spike (e.g., festive seasons or end-of-quarter boosts)

Low-sales months (e.g., off-season dips)

2. Customer Gender Distribution (Count Plot)
Insight: Shows the gender breakdown of the customer base.

If skewed, it could indicate a target audience imbalance.

Example: If more males are buying, tailor ads or offers to attract more female customers.

3. Sales by Product Category (Bar Chart)
Insight: Visualizes total revenue from each product category.

Top-performing categories (e.g., "Electronics" or "Furniture")

Underperforming categories that may need promotion or product redesign.

4. Sales Over Time (Line Plot)
Insight: Shows daily or transactional sales performance.

Useful for: Spotting trends, anomalies (e.g., sudden drops), Evaluating marketing campaign impacts, Detecting weekly/daily sales cycles

5. Correlation Heatmap
Insight: Reveals how numerical variables relate to each other.

Examples: Strong positive correlation between Quantity Sold and Total Amount (expected), Weak correlation between Discount and Total Amount may suggest ineffective discounting strategy

## EDA-Based Actionable Recommendations
1. Improve Inventory Based on Top Categories
Invest more in categories showing high total sales.

Consider phasing out or repositioning underperforming categories.

2. Reward Top Customers
Implement loyalty programs for the top 10 spending customers to retain them.

Use targeted promotions or exclusive offers.

3. Gender-Specific Campaigns
If there's a gender imbalance in the customer base:

Run promotions that appeal to the underrepresented group.

Customize product bundles and marketing based on dominant gender patterns.

4. Capitalize on High Sales Months
Analyze what contributed to sales spikes in specific months.

Align future campaigns and stock planning around those periods.

5. Analyze Low-Sales Periods
Use additional factors (e.g., weather, holidays, marketing gaps) to understand low-performance periods.

Run discounts or clearance sales during dips to maintain cash flow.

## How to Run
1. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```
2. Run the Python script (sales_analysis.py)
3. Ensure the dataset path is correct
