import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv(r"C:\Users\anjan\Downloads\apps.csv")
df.head()
df.info()
df = df.dropna(subset=['App', 'Category'])
print(df.shape)

# Drop Unnamed column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert 'Reviews' to numeric
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Clean 'Installs'
df['Installs'] = df['Installs'].str.replace('+', '', regex=False).str.replace(',', '', regex=False)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Clean 'Price'
df['Price'] = df['Price'].str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

def clean_size(value):
    if isinstance(value, str):
        value = value.strip()
        if 'M' in value:
            return float(value.replace('M', ''))
        elif 'k' in value:
            return float(value.replace('k', '')) / 1024  # convert kB to MB
        elif value == 'Varies with device':
            return np.nan
    return value  # leave float values untouched

# Apply cleaning if not already applied
df['Size'] = df['Size'].apply(clean_size)

# Ensure float type
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

df.dropna(subset=['Size', 'Rating'], inplace=True)


# Category Distribution
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='barh', color='teal')
plt.title("Number of Apps per Category")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=20, kde=True, color='orange')
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Average Rating by Category
avg_rating = df.groupby('Category')['Rating'].mean().sort_values()
avg_rating.plot(kind='barh', figsize=(10, 8), color='purple', title="Average Rating by Category")
plt.xlabel("Average Rating")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Scatter plot of Size vs Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Size', y='Rating', hue='Category', data=df)
plt.title("App Size vs Rating")
plt.xlabel("Size (MB)")
plt.ylabel("Rating")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# Free vs Paid Apps
plt.figure(figsize=(8, 6))
df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, title='Free vs Paid Apps', colors=['green', 'lightblue'])
plt.ylabel('')
plt.show()

# Top Installed Apps
top_apps = df.sort_values(by='Installs', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Installs', y='App', data=top_apps, palette='crest')
plt.title('Top 10 Installed Apps')
plt.show()

# Estimating Revenue from Paid Apps
paid_apps = df[df['Type'] == 'Paid'].copy()
paid_apps['Revenue'] = paid_apps['Price'] * paid_apps['Installs']
top_revenue = paid_apps.sort_values(by='Revenue', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Revenue', y='App', data=top_revenue, palette='flare')
plt.title('Top 10 Revenue Generating Paid Apps')
plt.show()

# Sentiment Analysis of User Reviews
# Load user reviews dataset
reviews = pd.read_csv(r"C:\Users\anjan\Downloads\user_reviews.csv")
reviews.dropna(subset=['Translated_Review'], inplace=True)
print(reviews.shape)
print(reviews.info())

# Add polarity
reviews['Polarity'] = reviews['Translated_Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Classify Sentiment
reviews['Sentiment'] = reviews['Polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Plot sentiment distribution
plt.figure(figsize=(6, 4))
reviews['Sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("User Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Interactive Plot with Plotly
fig = px.scatter(df, x='Size', y='Rating', color='Category', hover_data=['App'],
                 title='Interactive Plot: App Size vs Rating by Category')

fig.show()
