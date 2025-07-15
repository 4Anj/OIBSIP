# YouTube Video Categories (Canada) – JSON Analysis
This project involves loading and analyzing a nested JSON file (CA_category_id.json) containing YouTube video category metadata specific to Canada. The analysis focuses on extracting relevant details, cleaning the structure, and visualizing assignable category types.

### Dataset
File Used: CA_category_id.json   https://www.kaggle.com/datasets/datasnaek/youtube-new

### Contents of the JSON:
Each item in the dataset contains:

id: Unique category ID

snippet.title: Category name

snippet.assignable: Whether this category is assignable to a video

snippet.channelId: Channel associated with the category

### Tools & Libraries
```bash
pandas
matplotlib
seaborn
json
```

### Workflow Summary
1. JSON Parsing & Normalization
Loaded and flattened nested JSON using pandas.json_normalize()

2. Data Cleaning
Standardized column names to lowercase, no spaces

Renamed columns for clarity:

id → category_id

snippet.title → category_title

snippet.assignable → is_assignable

snippet.channelId → channel_id

Dropped irrelevant fields like kind and etag

Removed duplicates

## Visualization
### Assignable vs Non-Assignable Categories
Used a countplot to show how many categories are assignable to videos.

This helps identify which categories are directly usable when uploading videos.

```python
sns.countplot(x='is_assignable', data=df)
```

### Key Insights
Most categories are assignable, meaning they can be used when uploading content.

Non-assignable categories might be internal or legacy categories maintained by YouTube.

### Assignable Category Counts (Sample):
```graphql
True     XX categories
False     X categories
```

### requirements.txt
```nginx
pandas
matplotlib
seaborn
```
Install dependencies using:

```bash
pip install -r requirements.txt
```
