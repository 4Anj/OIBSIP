# Customer Segmentation using RFM Analysis & K-Means Clustering
This project performs customer segmentation using RFM analysis (Recency, Frequency, Monetary) and K-Means clustering to identify behavior-based customer groups from a marketing dataset.

## Dataset
Filename: ifood_df.csv  https://www.kaggle.com/code/analystoleksandra/marketing-analytics-customer-segmentation

The dataset contains demographic and purchase-related information for customers of a retail brand.

Relevant columns used:

Recency: Days since last purchase

NumWebPurchases, NumCatalogPurchases, NumStorePurchases: Used to calculate Frequency

MntTotal: Total amount spent (Monetary)

### Tools and Libraries Used
pandas – Data manipulation

scikit-learn – Preprocessing and clustering

matplotlib & seaborn – Visualizations

### Steps Performed
1. Data Exploration & Cleaning
Checked data types, missing values, and summary statistics.

No major cleaning required for RFM fields.

2. RFM Feature Engineering
Recency: Days since last interaction

Frequency: Total purchase count across web, catalog, and store

Monetary: Total spend (MntTotal)

```python
rfm["Frequency"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
```

3. Data Scaling
Used StandardScaler to normalize the RFM values for better clustering performance.

4. Optimal Clusters – Elbow Method
Used the Within-Cluster Sum of Squares (WCSS) to identify the optimal number of clusters (k).

Elbow observed at k = 4, which was used for segmentation.

```python
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
```

5. 🤖 Customer Segmentation (K-Means Clustering)
Cluster labels assigned to each customer.

Segments visualized with scatterplots and bar charts.

### Visualizations
🔹 Elbow Method Plot
Helps determine the optimal number of clusters.

🔹 Scatter Plot (Recency vs Monetary)
Visualizes customer clusters by recency and total spend.

🔹 Bar Plot (Average RFM by Cluster)
Compares average values of Recency, Frequency, and Monetary for each cluster.

🔍 Insights & Cluster Meaning
| Cluster | Recency | Frequency | Monetary | Insight                                                   |
| ------- | ------- | --------- | -------- | --------------------------------------------------------- |
| 0       | High    | Low       | Low      | Dormant customers – long time since last purchase         |
| 1       | Low     | High      | High     | Loyal/high-value customers – frequent and recent spenders |
| 2       | Medium  | Medium    | Medium   | Potential regulars – show moderate activity               |
| 3       | Low     | Low       | Low      | New or low-engagement customers                           |


### Recommendations
Cluster 1: Reward loyal customers with exclusive deals or memberships.

Cluster 0: Re-engage dormant users via personalized offers or win-back campaigns.

Cluster 2: Upsell or cross-sell to push them toward loyalty.

Cluster 3: Nurture with onboarding emails or introductory offers.
