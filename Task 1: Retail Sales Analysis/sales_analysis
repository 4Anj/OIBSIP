import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r"C:\Users\anjan\OneDrive\Desktop\EcoTrack\retail_sales_dataset.xlsx")
df.head()

#1. Data cleaning

#check shape and inf
print(df.shape)
print(df.info())

# check missing values
print(df.isnull().sum())

print(df.dtypes)

# Convert date column
df["Date"]=pd.to_datetime(df["Date"])

#2. Statistics
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n",df.median(numeric_only=True))
print("\nMode:\n", df.mode().iloc[0])
print("\nStandard deviation:\n", df.std(numeric_only=True))

#3. time series analysis

#Group sales by month/year
df["Month"]=df["Date"].dt.month
df["Year"]=df["Date"].dt.year

df.columns = df.columns.str.strip()

monthly_sales=df.groupby(["Year","Month"])["Total Amount"].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_sales, x=monthly_sales.index, y="Total Amount")
plt.title("Monthly sales trend")
plt.xlabel("Time (Months)")
plt.ylabel("Total Sales")
plt.show()

# 4. customer and product analysis
# Top 10 customers by total sales
top_customers = df.groupby("Customer ID")["Total Amount"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Total Sales:\n", top_customers)

# Top 10 products by total sales
top_products = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Total Sales:\n", top_products)

# Customer Demographics
sns.countplot(x="Gender", data=df)
plt.title("Customer Gender Distribution")

# 5. Visualizations
# Bar Chart for product categories or regions
plt.figure(figsize=(10, 6))
df.groupby('Product Category')['Total Amount'].sum().sort_values().plot(kind='barh')
plt.title("Sales by Category")
plt.xlabel("Total Sales")
plt.ylabel("Category")
plt.show()

#line plot for sales over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Total Amount")
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# Heatmap for correlation between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

