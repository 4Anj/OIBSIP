import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\anjan\Downloads\AB_NYC_2019.csv")
print(df.head(10))

print(df.describe())
df.info()
print(df.shape)

print(df.isnull().sum())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.shape)

df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')
print(df.columns)

df['neighbourhood_group'] = df['neighbourhood_group'].str.title()
df['neighbourhood'] = df['neighbourhood'].str.title()
df['room_type'] = df['room_type'].str.title()
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['price'] = df['price'].astype(float)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
print(df.head(10))
df.info()

#before removing outliers
print("Shape before removing outliers:", df.shape)

#price boxplot 
plt.figure(figsize=(10, 4))
sns.boxplot(x='price', data=df)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.show()

#minimum nights boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x='minimum_nights', data=df)
plt.title('Minimum Nights Distribution')
plt.xlabel('Minimum Nights')
plt.show()

#removing outliers in 'price' column using IQR method
# Calculate IQR for 'price' column
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Keep only non-outlier rows
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
df = df[df['minimum_nights'] <= 365]

# Display the shape of the DataFrame after removing outliers
print(f"Shape after removing outliers: {df.shape}")
df.describe()

# After removing outliers
# Price boxplot after removing outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x='price', data=df)
plt.title('Price Distribution After Removing Outliers')
plt.xlabel('Price')
plt.show()

# Minimum nights boxplot after removing outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x='minimum_nights', data=df)
plt.title('Minimum Nights Distribution After Removing Outliers')
plt.xlabel('Minimum Nights')
plt.show()
