import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

with open(r"C:\Users\anjan\Downloads\CA_category_id.json") as f:
    data = json.load(f)
# Normalize nested JSON structure
df = pd.json_normalize(data['items'])

# Clean column names: lowercase, no spaces
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Print structure
print(df.head())
print(df.info())
df = df.drop(columns=['kind', 'etag'])

# Rename columns for clarity
df = df.rename(columns={
    'id': 'category_id',
    'snippet.channelid': 'channel_id',
    'snippet.title': 'category_title',
    'snippet.assignable': 'is_assignable'
})

# Optional: reorder
df = df[['category_id', 'category_title', 'is_assignable', 'channel_id']]
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

# Visualize the distribution of assignable vs non-assignable categories
sns.countplot(x='is_assignable', data=df)
plt.title("Assignable vs Non-Assignable Video Categories")
plt.xlabel("Is Assignable?")
plt.ylabel("Number of Categories")
plt.show()

# Count the number of assignable and non-assignable categories
assignable_counts = df.groupby('is_assignable')['category_title'].count()
print(assignable_counts)
