# Menu Nutrition Analysis using Python
This project performs an in-depth nutritional analysis of a fast-food restaurant's menu using Python. The goal is to explore caloric and sugar content across categories, identify the highest and lowest items, and generate meaningful visual insights.

## Dataset
Filename: menu.csv
https://www.kaggle.com/datasets/mcdonalds/nutrition-facts

The dataset contains nutritional values of food and beverages served at a restaurant.

Key columns: Item  Category  Calories  Sugars

Plus other nutritional components (e.g., Protein, Total Fat, Sodium, etc.)

## Tools & Libraries
pandas – Data manipulation

seaborn – Advanced visualizations

matplotlib – Plotting charts

## Features of the Analysis
1. Data Cleaning & Exploration
Removed column name whitespace

Checked null values and data types

Printed dataset shape, structure, and basic statistics

2. Descriptive Statistics
Calculated:

Mean

Median

Mode

Standard Deviation for numerical columns

## 3. Correlation Heatmap
Visualized relationships between nutritional values using a heatmap

4. Category-wise Analysis
Bar chart for average calories per food category

Bar chart for average sugar content per category

5. Top and Bottom Items

Top 10 high-calorie and low-calorie items

Top 10 high-sugar and low-sugar items

## Sample Visuals
Nutrient Correlation Heatmap

Average Calories per Category

Average Sugar per Category

## How to Run the Project
1. Install dependencies
```bash
pip install pandas matplotlib seaborn
```

2. Run the script
Make sure to adjust the dataset path in the code:
```python
df = pd.read_csv("menu.csv")
```
