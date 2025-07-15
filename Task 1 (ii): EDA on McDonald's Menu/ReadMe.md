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

## Visualization: Insights through Charts
### Heatmap – Nutrient Correlation
Relationships between numerical features (like Calories, Sugars, Total Fat, etc.).

Insight: Strong positive correlation between Calories and Total Fat, Sugars and Carbohydrates, etc.

```python
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
```

### Bar Chart – Average Calories per Category
Which food categories have the highest average calorie count.

Insight: Categories like Breakfast, Desserts, and Burgers have higher average calories.

```python
df.groupby('Category')['Calories'].mean().sort_values(ascending=False).plot(kind='barh')
```
### Bar Chart – Average Sugars per Category
Categories ranked by their sugar content.

Insight: Beverages and Desserts dominate in sugar content—critical for health-conscious choices.

```python
df.groupby('Category')['Sugars'].mean().sort_values(ascending=False).plot(kind='barh')
```

### Top/Bottom Items – Tables
Top 10 high-calorie and high-sugar items printed for transparency.

High-Calorie Items: Large burgers, shakes, combo meals.

Low-Calorie Items: Salads, apple slices, black coffee.

## Recommendations: Actionable Insights Based on EDA
Introduce Low-Sugar Options in Beverages

Beverages have extremely high sugar content.

Recommend unsweetened drinks, flavored waters, or diet options.

Reformulate High-Calorie Items

Items in the Breakfast and Burgers category contribute heavily to calories.

Consider smaller portions, lean protein options, or calorie labeling.

Promote Healthy Menu Items

Highlight low-calorie, high-nutrition items (e.g., salads, fruit sides).

Use menu labeling to guide calorie-conscious consumers.

Group Menu by Nutritional Tiers

Add icons or colors to menu items indicating “low-cal”, “low-sugar”, “high-protein” etc.

Run Customer Awareness Campaigns

Create awareness about sugar consumption and daily recommended limits.

Educate on how to build a balanced meal from your menu.

Reduce or modify underperforming high-calorie items.

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
