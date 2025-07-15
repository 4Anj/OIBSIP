import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df=pd.read_csv(r"C:\Users\anjan\Downloads\ifood_df.csv")
df.head()

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#descriptive statistics
rfm=pd.DataFrame()
rfm['Recency']=df["Recency"]
rfm["Frequency"]=(df["NumWebPurchases"]+df["NumCatalogPurchases"]+df["NumStorePurchases"])
rfm["Monetary"]=df["MntTotal"]
rfm.index.name="CustomerID"
print(rfm.describe())

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot WCSS
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments (Recency vs Monetary)')
plt.xlabel('Recency (days)')
plt.ylabel('Total Spend (₹)')
plt.show()


rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().plot(kind='bar', figsize=(10,6))
plt.title('Average RFM Scores by Cluster')
plt.xticks(rotation=0)
plt.legend(title='Metrics')
plt.xlabel('Cluster')
plt.ylabel('Value')
plt.show()
