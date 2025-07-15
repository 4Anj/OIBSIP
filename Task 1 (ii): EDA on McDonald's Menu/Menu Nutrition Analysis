import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\anjan\Downloads\menu.csv")
df.head()

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
df.columns=df.columns.str.strip()
df.dtypes

print("Mean:\n",df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n",df.mode().iloc[0])
print("Standard deviation:\n",df.std(numeric_only=True))

plt.figure(figsize=(14, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Nutrient Correlation Heatmap")
plt.show()

df.groupby('Category')['Calories'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title("Avg Calories per Category")
plt.show()

df.groupby('Category')['Sugars'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title("Avg Sugar Content per Category")
plt.show()

top_calories = df.sort_values(by='Calories', ascending=False).head(10)
low_calories = df.sort_values(by='Calories').head(10)

top_sugar = df.sort_values(by='Sugars', ascending=False).head(10)
low_sugar = df.sort_values(by='Sugars').head(10)
print("\nTop 10 High-Calorie Items:\n", top_calories[['Item', 'Calories']])
print("\nTop 10 Low-Calorie Items:\n", low_calories[['Item', 'Calories']])
print("\nTop 10 High-Sugar Items:\n", top_sugar[['Item', 'Sugars']])
print("\nTop 10 Low-Sugar Items:\n", low_sugar[['Item', 'Sugars']])
